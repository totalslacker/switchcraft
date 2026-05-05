// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Per-call deadline context threaded through the search pipeline.
///
/// Captures the instant `SwitchcraftStore.search()` was entered and the
/// effective deadline for that call. Passed explicitly through
/// `SearchEngine.searchHybrid` and into `SQLiteStorage` so every
/// enforcement point can compute `elapsed` against the same origin
/// and remaining budget against the same total deadline.
public struct SearchDeadlineContext: Sendable {
    /// The instant `SwitchcraftStore.search()` was entered.
    public let searchStart: ContinuousClock.Instant
    /// Total wall-time budget for the search call.
    public let deadline: Duration

    public init(searchStart: ContinuousClock.Instant, deadline: Duration) {
        self.searchStart = searchStart
        self.deadline = deadline
    }

    /// Elapsed time since `searchStart`.
    public var elapsed: Duration {
        ContinuousClock.now - searchStart
    }

    /// True when the deadline has been reached or exceeded.
    public var isExpired: Bool {
        elapsed >= deadline
    }
}
