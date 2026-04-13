// Interface targeting LibRaw 0.21+
#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

@interface LibRawBridge : NSObject
/// Extracts embedded JPEG preview only. Returns nil if no embedded preview exists.
/// Use for thumbnails — never triggers full RAW decode.
+ (nullable CGImageRef)previewAtPath:(NSString *)path CF_RETURNS_RETAINED;
/// Full decode: embedded JPEG preview first, falls back to full LibRaw decode.
/// Use for processing pipeline only, not thumbnails.
+ (nullable CGImageRef)decodeFileAtPath:(NSString *)path CF_RETURNS_RETAINED;
@end
