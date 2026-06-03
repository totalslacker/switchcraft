// SPDX-License-Identifier: Apache-2.0
import Foundation

/// In-memory reference implementation of `SwitchcraftStorage`.
///
/// Useful for tests, fast prototypes, and as an executable specification of
/// the storage contract. Not intended for production use — everything lives
/// in RAM and is lost when the actor is deallocated.
///
/// Full-text search is implemented as a naive case-insensitive token-overlap
/// score so that the contract is satisfied without pulling in an FTS
/// dependency. Real backends are expected to use a proper BM25 index.
public actor InMemoryStorage: SwitchcraftStorage {
    private var isOpen = false

    private var documents: [String: DocumentRecord] = [:]
    private var chunksByHash: [String: ChunkRecord] = [:]
    private var chunksByID: [Int64: ChunkRecord] = [:]
    private var generations: [Int64: GenerationRecord] = [:]
    private var bucketsByGeneration: [Int64: [BucketRecord]] = [:]

    private var nextChunkID: Int64 = 1
    private var nextGenerationID: Int64 = 1
    private var nextBucketID: Int64 = 1

    private let ftsTitleWeight: Float

    public init(ftsTitleWeight: Float = 3.0) {
        self.ftsTitleWeight = ftsTitleWeight
    }

    // MARK: - Lifecycle

    public func open() async throws {
        isOpen = true
    }

    public func close() async throws {
        isOpen = false
    }

    public func clear() async throws {
        documents.removeAll()
        chunksByHash.removeAll()
        chunksByID.removeAll()
        generations.removeAll()
        bucketsByGeneration.removeAll()
        nextChunkID = 1
        nextGenerationID = 1
        nextBucketID = 1
    }

    // MARK: - Documents

    public func upsertDocument(_ document: DocumentRecord) async throws {
        documents[document.uuid] = document
    }

    public func deleteDocument(uuid: String) async throws {
        documents.removeValue(forKey: uuid)
    }

    public func document(uuid: String) async throws -> DocumentRecord? {
        documents[uuid]
    }

    public func documents(matching filter: StorageFilter) async throws -> [DocumentRecord] {
        documents.values.filter { filter.matches($0) }
    }

    public func documents(forChunkHash hash: String) async throws -> [DocumentRecord] {
        documents.values.filter { $0.hash == hash }
    }

    public func documentCount() async throws -> Int {
        documents.count
    }

    public func indexedURLs() async throws -> Set<String> {
        Set(documents.keys)
    }

    // MARK: - Chunks

    public func upsertChunk(_ chunk: ChunkRecord) async throws -> ChunkRecord {
        if let existing = chunksByHash[chunk.hash] {
            return existing
        }
        var inserted = chunk
        inserted.id = nextChunkID
        nextChunkID += 1
        chunksByHash[inserted.hash] = inserted
        chunksByID[inserted.id] = inserted
        return inserted
    }

    public func chunk(hash: String) async throws -> ChunkRecord? {
        chunksByHash[hash]
    }

    public func chunk(id: Int64) async throws -> ChunkRecord? {
        chunksByID[id]
    }

    public func chunkCount() async throws -> Int {
        chunksByHash.count
    }

    // MARK: - Generations

    public func insertGeneration(_ generation: GenerationRecord) async throws -> GenerationRecord {
        var inserted = generation
        inserted.id = nextGenerationID
        nextGenerationID += 1
        generations[inserted.id] = inserted
        return inserted
    }

    public func generations() async throws -> [GenerationRecord] {
        generations.values.sorted { $0.id < $1.id }
    }

    public func deleteGeneration(id: Int64) async throws {
        generations.removeValue(forKey: id)
        bucketsByGeneration.removeValue(forKey: id)
    }

    public func updateGenerationEmbeddingCount(id: Int64, count: Int) async throws {
        generations[id]?.numEmbeddings = count
    }

    public func replaceGeneration(
        losingGenerationID: Int64,
        survivingRecord: GenerationRecord,
        survivingBuckets: [BucketRecord]
    ) async throws -> GenerationRecord? {
        // Delete the losing generation and all its buckets first.
        generations.removeValue(forKey: losingGenerationID)
        bucketsByGeneration.removeValue(forKey: losingGenerationID)

        // If no survivors, fully pruned — nothing to insert.
        guard !survivingBuckets.isEmpty else { return nil }

        // Insert the surviving generation with a fresh id.
        var newGen = survivingRecord
        newGen.id = nextGenerationID
        nextGenerationID += 1
        generations[newGen.id] = newGen

        // Insert each surviving bucket, assigning fresh ids and the new genID.
        for var bucket in survivingBuckets {
            bucket.id = nextBucketID
            nextBucketID += 1
            bucket.generationID = newGen.id
            bucketsByGeneration[newGen.id, default: []].append(bucket)
        }
        return newGen
    }

    // MARK: - Buckets

    public func insertBucket(_ bucket: BucketRecord) async throws -> BucketRecord {
        var inserted = bucket
        inserted.id = nextBucketID
        nextBucketID += 1
        bucketsByGeneration[inserted.generationID, default: []].append(inserted)
        return inserted
    }

    public func buckets(forGeneration generationID: Int64) async throws -> [BucketRecord] {
        bucketsByGeneration[generationID] ?? []
    }

    // MARK: - Full-text Search

    public func searchFullText(
        query: String,
        limit: Int,
        filter: StorageFilter
    ) async throws -> [FullTextHit] {
        let queryTerms = Self.tokenize(query)
        guard !queryTerms.isEmpty, limit > 0 else { return [] }

        var hits: [FullTextHit] = []
        for document in documents.values where filter.matches(document) {
            let bodyTerms = Self.tokenize(document.body)
            let bodySet = Set(bodyTerms)
            let bodyOverlap = queryTerms.reduce(into: 0) { count, term in
                if bodySet.contains(term) { count += 1 }
            }

            var titleOverlap = 0
            if let title = document.title {
                let titleSet = Set(Self.tokenize(title))
                titleOverlap = queryTerms.reduce(into: 0) { count, term in
                    if titleSet.contains(term) { count += 1 }
                }
            }

            let rawScore = Float(bodyOverlap) + Float(titleOverlap) * ftsTitleWeight
            if rawScore > 0 {
                let score = rawScore / Float(queryTerms.count)
                hits.append(FullTextHit(uuid: document.uuid, score: score))
            }
        }

        // Total order: higher score first, ties broken by uuid ascending.
        hits.sort { lhs, rhs in
            if lhs.score != rhs.score { return lhs.score > rhs.score }
            return lhs.uuid < rhs.uuid
        }
        return Array(hits.prefix(limit))
    }

    private static func tokenize(_ text: String) -> [String] {
        text.lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
            .filter { !$0.isEmpty }
    }
}
