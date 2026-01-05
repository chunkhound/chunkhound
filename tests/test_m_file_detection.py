"""Tests for .m file language detection (Objective-C vs MATLAB)."""

import tempfile
from pathlib import Path

from chunkhound.core.models.file import File
from chunkhound.core.types.common import Language


class TestMFileDetection:
    """Test suite for disambiguating .m files between Objective-C and MATLAB."""

    def test_detects_objc_interface(self):
        """Test that @interface marker correctly identifies Objective-C files."""
        objc_code = """
#import <Foundation/Foundation.h>

@interface MyClass : NSObject
@property (nonatomic, strong) NSString *name;
- (void)doSomething;
@end
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(objc_code)
            temp_path = Path(f.name)

        try:
            language = File._detect_language(temp_path)
            assert language == Language.OBJC, (
                "File with @interface should be detected as Objective-C"
            )
        finally:
            temp_path.unlink()

    def test_detects_objc_implementation(self):
        """Test that @implementation marker correctly identifies Objective-C files."""
        objc_code = """
#import "MyClass.h"

@implementation MyClass

- (void)doSomething {
    NSLog(@"Doing something");
}

@end
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(objc_code)
            temp_path = Path(f.name)

        try:
            language = File._detect_language(temp_path)
            assert language == Language.OBJC, (
                "File with @implementation should be detected as Objective-C"
            )
        finally:
            temp_path.unlink()

    def test_detects_objc_protocol(self):
        """Test that @protocol marker correctly identifies Objective-C files."""
        objc_code = """
#import <Foundation/Foundation.h>

@protocol MyProtocol <NSObject>
@required
- (void)requiredMethod;
@optional
- (void)optionalMethod;
@end
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(objc_code)
            temp_path = Path(f.name)

        try:
            language = File._detect_language(temp_path)
            assert language == Language.OBJC, (
                "File with @protocol should be detected as Objective-C"
            )
        finally:
            temp_path.unlink()

    def test_detects_objc_class_forward_declaration(self):
        """Test that @class marker correctly identifies Objective-C files."""
        objc_code = """
@class MyOtherClass;

@interface MyClass : NSObject
- (void)workWith:(MyOtherClass *)obj;
@end
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(objc_code)
            temp_path = Path(f.name)

        try:
            language = File._detect_language(temp_path)
            assert language == Language.OBJC, (
                "File with @class should be detected as Objective-C"
            )
        finally:
            temp_path.unlink()

    def test_detects_objc_import(self):
        """Test that #import convention correctly identifies Objective-C files."""
        objc_code = """
#import <UIKit/UIKit.h>
#import "MyClass.h"

// Some Objective-C code
int main(int argc, char *argv[]) {
    return 0;
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(objc_code)
            temp_path = Path(f.name)

        try:
            language = File._detect_language(temp_path)
            assert language == Language.OBJC, (
                "File with #import should be detected as Objective-C"
            )
        finally:
            temp_path.unlink()

    def test_detects_matlab_function(self):
        """Test that MATLAB function syntax correctly identifies MATLAB files."""
        matlab_code = """
function result = calculateSum(a, b)
    % This is a MATLAB function
    result = a + b;
end
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(matlab_code)
            temp_path = Path(f.name)

        try:
            language = File._detect_language(temp_path)
            assert language == Language.MATLAB, (
                "File with function keyword should be detected as MATLAB"
            )
        finally:
            temp_path.unlink()

    def test_defaults_to_matlab_for_empty_file(self):
        """Test that empty .m files default to MATLAB (historical precedent)."""
        matlab_code = ""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(matlab_code)
            temp_path = Path(f.name)

        try:
            language = File._detect_language(temp_path)
            assert language == Language.MATLAB, "Empty .m file should default to MATLAB"
        finally:
            temp_path.unlink()

    def test_defaults_to_matlab_for_comment_only(self):
        """Test that .m files with only comments default to MATLAB."""
        matlab_code = """
% This is a comment
% MATLAB uses % for comments
% No code here yet
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(matlab_code)
            temp_path = Path(f.name)

        try:
            language = File._detect_language(temp_path)
            assert language == Language.MATLAB, (
                "Comment-only .m file should default to MATLAB"
            )
        finally:
            temp_path.unlink()

    def test_mm_extension_not_detected(self):
        """Test that .mm files return None (don't need content detection)."""
        # .mm is unambiguous (Objective-C++)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mm", delete=False) as f:
            f.write("// Objective-C++ code")
            temp_path = Path(f.name)

        try:
            language = File._detect_language(temp_path)
            assert language is None, (
                ".mm files should return None (use extension-based detection)"
            )
        finally:
            temp_path.unlink()

    def test_non_m_extension_returns_none(self):
        """Test that non-.m files return None (don't need content detection)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello')")
            temp_path = Path(f.name)

        try:
            language = File._detect_language(temp_path)
            assert language is None, "Non-.m files should return None"
        finally:
            temp_path.unlink()

    def test_handles_unreadable_file_gracefully(self):
        """Test that detection handles file read errors gracefully."""
        # Create a path that doesn't exist
        nonexistent_path = Path("/tmp/nonexistent_file_xyz123.m")

        # Should return MATLAB as fallback, not raise an exception
        language = File._detect_language(nonexistent_path)
        assert language == Language.MATLAB, (
            "Unreadable .m file should default to MATLAB"
        )

    def test_detects_objc_within_first_1kb(self):
        """Test that Objective-C markers within first 1KB are detected."""
        # Create a file with @interface appearing within the first 1024 bytes
        objc_code = """
// Copyright notice and comments that take up space
// Line 3
// Line 4
// Line 5

#import <Foundation/Foundation.h>

@interface MyClass : NSObject
- (void)method;
@end
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(objc_code)
            temp_path = Path(f.name)

        try:
            language = File._detect_language(temp_path)
            assert language == Language.OBJC, (
                "Objective-C markers within 1KB should be detected"
            )
        finally:
            temp_path.unlink()

    def test_matlab_script_without_function(self):
        """Test that MATLAB scripts (without function keyword) default to MATLAB."""
        matlab_code = """
% MATLAB script
x = 1:10;
y = x.^2;
plot(x, y);
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write(matlab_code)
            temp_path = Path(f.name)

        try:
            language = File._detect_language(temp_path)
            assert language == Language.MATLAB, "MATLAB script should default to MATLAB"
        finally:
            temp_path.unlink()
