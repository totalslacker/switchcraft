// SPDX-License-Identifier: Apache-2.0
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Executes a block inside an Objective-C @try/@catch guard and returns
/// any caught NSException, or nil if the block completed without throwing.
///
/// Swift's do-catch cannot intercept Objective-C exceptions. This bridge
/// converts them into a returnable value so the Swift caller can translate
/// them into typed Swift errors instead of letting them propagate to
/// std::terminate. See ADR 018.
@interface MLExceptionCatcher : NSObject

+ (nullable NSException *)performBlock:(NS_NOESCAPE void (^)(void))block;

@end

NS_ASSUME_NONNULL_END
