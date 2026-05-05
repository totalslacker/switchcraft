// SPDX-License-Identifier: Apache-2.0
#import "MLExceptionCatcher.h"

@implementation MLExceptionCatcher

+ (nullable NSException *)performBlock:(NS_NOESCAPE void (^)(void))block {
    @try {
        block();
    } @catch (NSException *exception) {
        return exception;
    }
    return nil;
}

@end
