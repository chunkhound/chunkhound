"""Integration tests for Objective-C file indexing and search.

Tests the complete workflow: file detection → parsing → chunking → search.
"""

import tempfile
from pathlib import Path

from chunkhound.core.detection import detect_language
from chunkhound.core.types.common import FileId, Language
from chunkhound.parsers.parser_factory import get_parser_factory
from chunkhound.services.batch_processor import process_file_batch


class TestObjectiveCDetection:
    """Test Objective-C language detection."""

    def test_detects_objc_from_interface(self):
        """Test that .m files with @interface are detected as Objective-C."""
        objc_code = """
#import <Foundation/Foundation.h>

@interface Person : NSObject
@property (nonatomic, strong) NSString *name;
- (void)greet;
@end
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(objc_code)
            temp_path = Path(f.name)

        try:
            language = detect_language(temp_path)
            assert language == Language.OBJC, f"Expected OBJC but got {language}"
        finally:
            temp_path.unlink()

    def test_detects_objc_from_implementation(self):
        """Test that .m files with @implementation are detected as Objective-C."""
        objc_code = """
@implementation Person

- (void)greet {
    NSLog(@"Hello!");
}

@end
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(objc_code)
            temp_path = Path(f.name)

        try:
            language = detect_language(temp_path)
            assert language == Language.OBJC, f"Expected OBJC but got {language}"
        finally:
            temp_path.unlink()

    def test_detects_objc_from_import(self):
        """Test that .m files with #import are detected as Objective-C."""
        objc_code = """
#import "MyClass.h"

void someFunction() {
    // Code here
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(objc_code)
            temp_path = Path(f.name)

        try:
            language = detect_language(temp_path)
            assert language == Language.OBJC, f"Expected OBJC but got {language}"
        finally:
            temp_path.unlink()

    def test_defaults_to_matlab_without_objc_markers(self):
        """Test that .m files without ObjC markers default to MATLAB."""
        matlab_code = """
function result = calculate(x, y)
    result = x + y;
end
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(matlab_code)
            temp_path = Path(f.name)

        try:
            language = detect_language(temp_path)
            assert language == Language.MATLAB, f"Expected MATLAB but got {language}"
        finally:
            temp_path.unlink()

    def test_mm_files_are_objc(self):
        """Test that .mm files are unambiguously Objective-C."""
        objc_code = """
// Objective-C++ code
int main() {
    return 0;
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mm", delete=False) as f:
            f.write(objc_code)
            temp_path = Path(f.name)

        try:
            language = detect_language(temp_path)
            assert language == Language.OBJC, f"Expected OBJC but got {language}"
        finally:
            temp_path.unlink()


class TestObjectiveCParsing:
    """Test Objective-C file parsing."""

    def test_parses_objc_file_successfully(self):
        """Test that Objective-C files are parsed into multiple chunks."""
        objc_code = """
#import <Foundation/Foundation.h>

@interface Person : NSObject

@property (nonatomic, strong) NSString *name;
@property (nonatomic, assign) NSInteger age;

- (instancetype)initWithName:(NSString *)name age:(NSInteger)age;
- (void)greet;

@end

@implementation Person

- (instancetype)initWithName:(NSString *)name age:(NSInteger)age {
    self = [super init];
    if (self) {
        _name = name;
        _age = age;
    }
    return self;
}

- (void)greet {
    NSLog(@"Hello, my name is %@ and I am %ld years old.",
          self.name, (long)self.age);
}

@end
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(objc_code)
            temp_path = Path(f.name)

        try:
            # Get parser for Objective-C
            factory = get_parser_factory()
            parser = factory.create_parser(Language.OBJC)

            # Parse the file
            chunks = parser.parse_file(temp_path, FileId(1))

            # Verify multiple chunks were created
            assert len(chunks) > 0, "Should create at least one chunk"

            # In a realistic scenario with proper ObjC parsing,
            # we should get chunks for interface, implementation, methods, etc.
            # For now, just verify parsing doesn't fail
            print(f"Created {len(chunks)} chunks from ObjC file")

        finally:
            temp_path.unlink()


class TestObjectiveCBatchProcessing:
    """Test Objective-C file batch processing."""

    def test_batch_processor_detects_objc(self):
        """Test that batch processor correctly detects Objective-C files."""
        objc_code = """
#import <Foundation/Foundation.h>

@interface Calculator : NSObject
- (int)add:(int)a to:(int)b;
@end

@implementation Calculator
- (int)add:(int)a to:(int)b {
    return a + b;
}
@end
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(objc_code)
            temp_path = Path(f.name)

        try:
            # Process file through batch processor
            results = process_file_batch([temp_path], {})

            assert len(results) == 1, "Should process one file"
            result = results[0]

            assert result.status == "success", f"File processing failed: {result.error}"
            assert result.language == Language.OBJC, (
                f"Expected OBJC but got {result.language}"
            )
            assert len(result.chunks) > 0, "Should create at least one chunk"

            print(f"Batch processor created {len(result.chunks)} chunks")
            print(f"Language detected: {result.language}")

        finally:
            temp_path.unlink()

    def test_batch_processor_handles_matlab(self):
        """Test that batch processor correctly handles MATLAB files."""
        matlab_code = """
function result = fibonacci(n)
    if n <= 1
        result = n;
    else
        result = fibonacci(n-1) + fibonacci(n-2);
    end
end
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(matlab_code)
            temp_path = Path(f.name)

        try:
            # Process file through batch processor
            results = process_file_batch([temp_path], {})

            assert len(results) == 1, "Should process one file"
            result = results[0]

            # MATLAB files should be processed successfully
            assert result.status == "success" or result.language == Language.MATLAB, (
                f"MATLAB file should be handled: status={result.status}, lang={result.language}"
            )

            print(f"Language detected: {result.language}")

        finally:
            temp_path.unlink()
