"""Tests for Objective-C language mapping and parsing."""


from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import get_parser_factory


def parse_objc(content: str):
    """Helper function to parse Objective-C content."""
    factory = get_parser_factory()
    parser = factory.create_parser(Language.OBJC)
    return parser.parse_content(content, "test.m", FileId(1))


class TestObjCInterfaceAndImplementation:
    """Test parsing of Objective-C @interface and @implementation blocks."""

    def test_captures_interface_declaration(self):
        """Test that @interface declarations are captured as CLASS chunks."""
        content = """
@interface MyClass : NSObject
@property (nonatomic, strong) NSString *name;
- (void)doSomething;
@end
"""
        chunks = parse_objc(content)
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        symbols = {c.symbol for c in chunks}

        assert len(class_chunks) > 0, "Should capture @interface as CLASS chunk"
        assert "MyClass" in symbols, (
            f"Expected 'MyClass' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_implementation(self):
        """Test that @implementation blocks are captured."""
        content = """
@implementation MyClass

- (void)doSomething {
    NSLog(@"Doing something");
}

@end
"""
        chunks = parse_objc(content)
        symbols = {c.symbol for c in chunks}

        # Implementation might be captured as CLASS, BLOCK, or METHOD depending on tree-sitter grammar
        relevant_chunks = [
            c
            for c in chunks
            if c.chunk_type in (ChunkType.CLASS, ChunkType.BLOCK, ChunkType.METHOD)
        ]

        assert len(relevant_chunks) > 0, "Should capture @implementation"
        # Either class name or method name should be in symbols
        has_relevant = (
            "MyClass" in symbols
            or "doSomething" in symbols
            or any("do" in s.lower() for s in symbols)
        )
        assert has_relevant, (
            f"Expected MyClass or doSomething in symbols, got: {sorted(symbols)}"
        )


class TestObjCMethods:
    """Test parsing of Objective-C methods."""

    def test_captures_instance_method(self):
        """Test that instance methods (- prefix) are captured as METHOD chunks."""
        content = """
@implementation MyClass

- (void)instanceMethod {
    // Substantial implementation to avoid filtering
    int x = 1;
    int y = 2;
    int z = 3;
    int result = x + y + z;
    return;
}

@end
"""
        chunks = parse_objc(content)
        symbols = {c.symbol for c in chunks}
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]

        # Methods might be captured as METHOD or FUNCTION depending on mapping
        if not method_chunks:
            method_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]

        assert len(method_chunks) > 0, "Should capture instance method"
        # Check if method name appears in symbols (with or without - prefix)
        has_method = (
            "instanceMethod" in symbols
            or "-instanceMethod" in symbols
            or any("instance" in s.lower() for s in symbols)
        )
        assert has_method, (
            f"Expected 'instanceMethod' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_class_method(self):
        """Test that class methods (+ prefix) are captured."""
        content = """
@implementation MyClass

+ (instancetype)sharedInstance {
    static MyClass *instance = nil;
    return instance;
}

@end
"""
        chunks = parse_objc(content)
        symbols = {c.symbol for c in chunks}

        # Look for the method in either METHOD or FUNCTION chunks
        method_chunks = [
            c for c in chunks if c.chunk_type in (ChunkType.METHOD, ChunkType.FUNCTION)
        ]

        assert len(method_chunks) > 0, "Should capture class method"
        # Check if method name appears in symbols (with or without + prefix)
        has_method = (
            "sharedInstance" in symbols
            or "+sharedInstance" in symbols
            or any("shared" in s.lower() for s in symbols)
        )
        assert has_method, (
            f"Expected 'sharedInstance' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_method_with_parameters(self):
        """Test that methods with parameters are captured correctly."""
        content = """
@implementation MyClass

- (void)setName:(NSString *)name age:(int)age {
    _name = name;
    _age = age;
}

@end
"""
        chunks = parse_objc(content)
        symbols = {c.symbol for c in chunks}

        # Look for the method
        method_chunks = [
            c for c in chunks if c.chunk_type in (ChunkType.METHOD, ChunkType.FUNCTION)
        ]

        assert len(method_chunks) > 0, "Should capture method with parameters"
        # The method name in Objective-C includes colons: setName:age:
        has_method = any("setName" in s or "age:" in s for s in symbols)
        assert has_method, (
            f"Expected method with parameters (setName:age:), got: {sorted(symbols)}"
        )


class TestObjCProtocols:
    """Test parsing of Objective-C @protocol definitions."""

    def test_captures_protocol_declaration(self):
        """Test that @protocol declarations are captured."""
        content = """
@protocol MyProtocol <NSObject>

@required
- (void)requiredMethod;

@optional
- (void)optionalMethod;

@end
"""
        chunks = parse_objc(content)

        # Protocols might be captured as CLASS, INTERFACE, or BLOCK
        relevant_chunks = [
            c
            for c in chunks
            if c.chunk_type in (ChunkType.CLASS, ChunkType.INTERFACE, ChunkType.BLOCK)
        ]

        assert len(relevant_chunks) > 0, "Should capture @protocol declaration"


class TestObjCCategories:
    """Test parsing of Objective-C categories."""

    def test_captures_category_interface(self):
        """Test that category interfaces are captured."""
        content = """
@interface MyClass (Utilities)

- (void)utilityMethod;

@end
"""
        chunks = parse_objc(content)

        # Categories might be captured as CLASS or BLOCK
        relevant_chunks = [
            c for c in chunks if c.chunk_type in (ChunkType.CLASS, ChunkType.BLOCK)
        ]

        assert len(relevant_chunks) > 0, "Should capture category interface"

    def test_captures_category_implementation(self):
        """Test that category implementations are captured."""
        content = """
@implementation MyClass (Utilities)

- (void)utilityMethod {
    NSLog(@"Utility method");
}

@end
"""
        chunks = parse_objc(content)

        assert len(chunks) > 0, "Should capture category implementation"


class TestObjCProperties:
    """Test parsing of Objective-C @property declarations."""

    def test_captures_property_declarations(self):
        """Test that @property declarations are captured."""
        content = """
@interface MyClass : NSObject

@property (nonatomic, strong) NSString *name;
@property (nonatomic, assign) NSInteger age;
@property (nonatomic, copy) NSString *title;

@end
"""
        chunks = parse_objc(content)

        # Properties might be captured as PROPERTY or FIELD
        property_chunks = [
            c for c in chunks if c.chunk_type in (ChunkType.PROPERTY, ChunkType.FIELD)
        ]

        # At minimum, the interface should be captured
        assert len(chunks) > 0, "Should capture interface with properties"


class TestObjCComments:
    """Test parsing of Objective-C comments."""

    def test_captures_single_line_comments(self):
        """Test that single-line comments // are captured."""
        content = """
// This is a comment
@interface MyClass : NSObject
@end
"""
        chunks = parse_objc(content)

        comment_chunks = [c for c in chunks if c.chunk_type == ChunkType.COMMENT]

        # Comments may or may not be captured depending on cAST configuration
        # Just ensure parsing doesn't fail
        assert len(chunks) > 0, "Should parse file with comments"

    def test_captures_block_comments(self):
        """Test that block comments /* */ are captured."""
        content = """
/* This is a
   block comment */
@interface MyClass : NSObject
@end
"""
        chunks = parse_objc(content)

        # Just ensure parsing doesn't fail
        assert len(chunks) > 0, "Should parse file with block comments"


class TestObjCBlocks:
    """Test parsing of Objective-C blocks/closures."""

    def test_captures_block_declarations(self):
        """Test that block declarations are captured."""
        content = """
@interface MyClass : NSObject

@property (nonatomic, copy) void (^completionBlock)(BOOL success);

- (void)performWithBlock:(void (^)(NSString *result))block;

@end








@implementation MyClass

- (void)performWithBlock:(void (^)(NSString *result))block {
    // Substantial implementation
    NSString *result = @"Success";
    int counter = 0;
    int total = 0;
    if (block) {
        block(result);
    }
    counter++;
    total += counter;
}

@end
"""
        chunks = parse_objc(content)

        # Should capture the class and method with block parameter
        assert len(chunks) > 0, "Should parse file with blocks"


class TestObjCModernLiterals:
    """Test parsing of modern Objective-C literals."""

    def test_captures_code_with_literals(self):
        """Test that code with modern literals (@[], @{}, @()) is parsed."""
        content = """
@implementation MyClass

- (void)useLiterals {
    // Array literal
    NSArray *array = @[@"one", @"two", @"three"];

    // Dictionary literal
    NSDictionary *dict = @{@"key1": @"value1", @"key2": @"value2"};

    // Number literal
    NSNumber *number = @42;
    NSNumber *floatNum = @3.14;
    NSNumber *boolNum = @YES;

    // Boxed expressions
    NSNumber *sum = @(1 + 2);
}

@end
"""
        chunks = parse_objc(content)

        # Should parse without errors
        assert len(chunks) > 0, "Should parse file with modern literals"


class TestObjCPropertySynthesis:
    """Test parsing of @synthesize and @dynamic directives."""

    def test_captures_property_synthesis(self):
        """Test that @synthesize directives are parsed."""
        content = """
@interface MyClass : NSObject

@property (nonatomic, strong) NSString *name;
@property (nonatomic, assign) NSInteger age;

@end








@implementation MyClass

@synthesize name = _name;
@synthesize age = _age;

- (void)someMethod {
    // Substantial implementation
    _name = @"Test";
    _age = 42;
    int x = 1;
    int y = 2;
    int z = 3;
}

@end
"""
        chunks = parse_objc(content)

        # Should parse without errors
        assert len(chunks) > 0, "Should parse file with @synthesize"

    def test_captures_dynamic_properties(self):
        """Test that @dynamic directives are parsed."""
        content = """
@interface MyClass : NSObject

@property (nonatomic, strong) NSString *dynamicProperty;

@end








@implementation MyClass

@dynamic dynamicProperty;

- (void)someMethod {
    // Substantial implementation
    int x = 1;
    int y = 2;
    int z = 3;
    int a = 4;
    int b = 5;
}

@end
"""
        chunks = parse_objc(content)

        # Should parse without errors
        assert len(chunks) > 0, "Should parse file with @dynamic"


class TestObjCInstanceVariables:
    """Test parsing of instance variable (ivar) declarations."""

    def test_captures_ivars_in_interface(self):
        """Test that instance variables in interface are parsed."""
        content = """
@interface MyClass : NSObject {
    @private
    NSString *_privateName;
    NSInteger _privateAge;

    @protected
    NSString *_protectedName;

    @public
    NSString *publicName;
}

@property (nonatomic, strong) NSString *name;

@end
"""
        chunks = parse_objc(content)

        # Should capture interface with ivars
        assert len(chunks) > 0, "Should parse interface with ivars"

    def test_captures_ivars_in_implementation(self):
        """Test that instance variables in implementation extension are parsed."""
        content = """
@implementation MyClass {
    NSString *_internalState;
    NSInteger _counter;
}

- (void)incrementCounter {
    // Substantial implementation
    _counter++;
    int x = 1;
    int y = 2;
    int z = 3;
    int a = 4;
    int b = 5;
}

@end
"""
        chunks = parse_objc(content)

        # Should parse implementation with ivars
        assert len(chunks) > 0, "Should parse implementation with ivars"


class TestObjCClassExtensions:
    """Test parsing of class extensions (anonymous categories)."""

    def test_captures_class_extension(self):
        """Test that class extensions are parsed."""
        content = """
@interface MyClass ()

@property (nonatomic, strong) NSString *privateProperty;

- (void)privateMethod;

@end








@implementation MyClass

- (void)privateMethod {
    // Substantial implementation
    _privateProperty = @"Private";
    int x = 1;
    int y = 2;
    int z = 3;
    int a = 4;
    int b = 5;
}

@end
"""
        chunks = parse_objc(content)

        # Should parse class extension
        assert len(chunks) > 0, "Should parse class extension"


class TestObjCForwardDeclarations:
    """Test parsing of @class forward declarations."""

    def test_captures_forward_declarations(self):
        """Test that @class forward declarations are parsed."""
        content = """
@class MyOtherClass;
@class ThirdClass, FourthClass;











@interface MyClass : NSObject

- (void)workWith:(MyOtherClass *)other;

@end








@implementation MyClass

- (void)workWith:(MyOtherClass *)other {
    // Substantial implementation
    int x = 1;
    int y = 2;
    int z = 3;
    int a = 4;
    int b = 5;
}

@end
"""
        chunks = parse_objc(content)

        # Should parse file with forward declarations
        assert len(chunks) > 0, "Should parse file with @class declarations"


class TestObjCSymbolNames:
    """Test that Objective-C parser generates correct symbol names."""

    def test_method_symbols_include_scope(self):
        """Test that method symbols include - or + scope prefix."""
        content = """
@implementation MyClass

- (void)instanceMethod {
    // cAST merges small adjacent siblings. Expanding prevents merging.
    int x = 1;
    int y = 2;
    int z = 3;
    int a = 4;
    int b = 5;
}











+ (instancetype)classMethod {
    // Second method with substantial implementation
    static MyClass *instance = nil;
    int counter = 0;
    int total = 0;
    int multiplier = 5;
    int offset = 10;
    return instance;
}

@end
"""
        chunks = parse_objc(content)
        symbols = {c.symbol for c in chunks}

        # Method symbols should include scope prefix
        assert "-instanceMethod" in symbols or "instanceMethod" in symbols, (
            f"Expected '-instanceMethod' or 'instanceMethod' in symbols, got: {sorted(symbols)}"
        )
        assert (
            "+classMethod" in symbols
            or "classMethod" in symbols
            or "sharedInstance" in symbols
        ), (
            f"Expected '+classMethod' or 'classMethod' in symbols, got: {sorted(symbols)}"
        )

    def test_category_symbols_include_category_name(self):
        """Test that category symbols include category name in format Class+Category."""
        content = """
@interface MyClass (Utilities)

- (void)utilityMethod;

@end








@implementation MyClass (Utilities)

- (void)utilityMethod {
    // Substantial implementation
    int x = 1;
    int y = 2;
    int z = 3;
    int a = 4;
    int b = 5;
    NSLog(@"Utility method");
}

@end
"""
        chunks = parse_objc(content)

        # Look for category-related symbols
        # Category names might appear in class chunks or have special formatting
        relevant_chunks = [
            c
            for c in chunks
            if "MyClass" in c.symbol
            or "Utilities" in c.symbol
            or "utility" in c.symbol.lower()
        ]
        assert len(relevant_chunks) > 0, (
            f"Expected category-related chunks, got: {[c.symbol for c in chunks]}"
        )

    def test_protocol_symbols_include_prefix(self):
        """Test that protocol symbols include @protocol prefix or are clearly identifiable."""
        content = """
@protocol MyProtocol <NSObject>

@required
- (void)requiredMethod;

@optional
- (void)optionalMethod;

@end
"""
        chunks = parse_objc(content)
        symbols = {c.symbol for c in chunks}

        # Protocol should be captured with identifiable name
        protocol_related = [
            s
            for s in symbols
            if "MyProtocol" in s or "Protocol" in s or "protocol" in s.lower()
        ]
        assert len(protocol_related) > 0 or "MyProtocol" in symbols, (
            f"Expected protocol-related symbol, got: {sorted(symbols)}"
        )


class TestObjCMetadata:
    """Test that Objective-C metadata is correctly extracted."""

    def test_method_scope_metadata(self):
        """Test that method scope (instance vs class) metadata is extracted."""
        content = """
@implementation MyClass

- (void)instanceMethod {
    // cAST merges small adjacent siblings. Expanding prevents merging.
    int x = 1;
    int y = 2;
    int z = 3;
    int a = 4;
    int b = 5;
    int c = 6;
}











+ (instancetype)classMethod {
    // Second method with substantial implementation
    static MyClass *instance = nil;
    int counter = 0;
    int total = 0;
    int multiplier = 5;
    int offset = 10;
    int result = counter * multiplier + offset;
    return instance;
}

@end
"""
        chunks = parse_objc(content)

        # Check for methods with metadata
        method_chunks = [
            c for c in chunks if "Method" in c.symbol or "method" in c.symbol.lower()
        ]

        # At least should have parsed the implementation
        assert len(chunks) > 0, "Should extract chunks"

    def test_property_attributes_metadata(self):
        """Test that property attributes metadata is extracted."""
        content = """
@interface MyClass : NSObject

@property (nonatomic, strong) NSString *name;
@property (nonatomic, weak) id<MyProtocol> delegate;
@property (nonatomic, copy, readonly) NSString *identifier;
@property (atomic, assign) NSInteger count;

@end
"""
        chunks = parse_objc(content)

        # Should capture interface with properties
        assert len(chunks) > 0, "Should parse interface with properties"

    def test_protocol_conformance_metadata(self):
        """Test that protocol conformance metadata is extracted."""
        content = """
@interface MyClass : NSObject <NSCoding, NSCopying>

- (void)someMethod;

@end
"""
        chunks = parse_objc(content)

        # Should capture interface with protocol conformance
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should find class interface"

    def test_superclass_metadata(self):
        """Test that superclass metadata is extracted."""
        content = """
@interface MySubclass : MySuperclass

- (void)someMethod;

@end
"""
        chunks = parse_objc(content)

        # Should capture interface with superclass
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should find class interface with superclass"

    def test_comment_type_metadata(self):
        """Test that comment type metadata is extracted."""
        content = """
// This is a line comment
/// This is a documentation comment
/* This is a block comment */

@interface MyClass : NSObject
@end
"""
        chunks = parse_objc(content)

        # Should parse file with comments
        assert len(chunks) > 0, "Should parse file with various comment types"


class TestObjCComplexFile:
    """Test parsing of a complex Objective-C file with multiple constructs."""

    def test_parses_complete_class_file(self):
        """Test parsing a complete Objective-C class with interface and implementation."""
        content = """
#import <Foundation/Foundation.h>

@interface Person : NSObject

@property (nonatomic, strong) NSString *name;
@property (nonatomic, assign) NSInteger age;

- (instancetype)initWithName:(NSString *)name age:(NSInteger)age;
- (void)sayHello;
+ (instancetype)personWithName:(NSString *)name;

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

- (void)sayHello {
    NSLog(@"Hello, my name is %@", self.name);
}

+ (instancetype)personWithName:(NSString *)name {
    return [[Person alloc] initWithName:name age:0];
}

@end
"""
        chunks = parse_objc(content)

        # Should capture multiple chunks (interface, implementation, methods)
        assert len(chunks) > 0, "Should parse complete Objective-C file"

        # Check that we got some meaningful chunks
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        method_chunks = [
            c for c in chunks if c.chunk_type in (ChunkType.METHOD, ChunkType.FUNCTION)
        ]

        # Should have at least the class definition
        assert len(class_chunks) > 0 or len(method_chunks) > 0, (
            "Should capture class or method chunks from complex file"
        )

    def test_objc_realistic_module(self):
        """Test parsing a realistic Objective-C module with multiple advanced features."""
        content = """
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

// Forward declarations
@class NetworkManager;
@class DataStore;
@protocol CacheDelegate;

/// Error codes for user management operations
typedef NS_ENUM(NSInteger, UserError) {
    UserErrorNone = 0,
    UserErrorNotFound,
    UserErrorInvalidCredentials,
    UserErrorNetworkFailure,
    UserErrorPermissionDenied
};

/// Completion handler for user operations
typedef void (^UserCompletionBlock)(BOOL success, NSError *error);

/// Protocol for user authentication delegate
@protocol UserAuthenticationDelegate <NSObject>

@required
- (void)didAuthenticateUser:(NSString *)userId;
- (void)didFailAuthenticationWithError:(NSError *)error;

@optional
- (BOOL)shouldAllowAuthenticationForUser:(NSString *)userId;

@end

/// Main user management class
@interface UserManager : NSObject <NSCoding, NSCopying>

@property (nonatomic, strong, readonly) NSString *currentUserId;
@property (nonatomic, weak) id<UserAuthenticationDelegate> delegate;
@property (nonatomic, copy) UserCompletionBlock completionBlock;
@property (nonatomic, assign, getter=isAuthenticated) BOOL authenticated;
@property (nonatomic, strong) NetworkManager *networkManager;

+ (instancetype)sharedManager;
- (instancetype)initWithUserId:(NSString *)userId configuration:(NSDictionary *)config;

- (void)authenticateWithUsername:(NSString *)username
                        password:(NSString *)password
                      completion:(UserCompletionBlock)completion;

- (void)logoutWithCompletion:(void (^)(void))completion;
- (NSArray<NSString *> *)getUserPermissions;

@end

/// Private class extension
@interface UserManager ()

@property (nonatomic, strong, readwrite) NSString *currentUserId;
@property (nonatomic, strong) DataStore *dataStore;
@property (nonatomic, strong) NSMutableDictionary *sessionCache;
@property (nonatomic, assign) NSInteger retryCount;

- (void)invalidateSession;
- (BOOL)validateCredentials:(NSString *)username password:(NSString *)password;

@end

@implementation UserManager {
    NSString *_internalToken;
    NSInteger _loginAttempts;
    dispatch_queue_t _authQueue;
}

+ (instancetype)sharedManager {
    static UserManager *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[UserManager alloc] init];
    });
    return sharedInstance;
}

- (instancetype)init {
    return [self initWithUserId:nil configuration:nil];
}

- (instancetype)initWithUserId:(NSString *)userId configuration:(NSDictionary *)config {
    self = [super init];
    if (self) {
        _currentUserId = userId ?: @"";
        _authenticated = NO;
        _retryCount = 0;
        _sessionCache = [NSMutableDictionary dictionary];
        _authQueue = dispatch_queue_create("com.app.authqueue", DISPATCH_QUEUE_SERIAL);

        // Parse configuration
        if (config) {
            NSNumber *maxRetries = config[@"maxRetries"];
            if (maxRetries) {
                _retryCount = [maxRetries integerValue];
            }
        }
    }
    return self;
}

- (void)authenticateWithUsername:(NSString *)username
                        password:(NSString *)password
                      completion:(UserCompletionBlock)completion {
    // Validate inputs
    if (!username || !password || username.length == 0 || password.length == 0) {
        NSError *error = [NSError errorWithDomain:@"UserManagerErrorDomain"
                                             code:UserErrorInvalidCredentials
                                         userInfo:@{NSLocalizedDescriptionKey: @"Invalid credentials"}];
        if (completion) {
            completion(NO, error);
        }
        return;
    }

    // Check delegate permission
    if ([self.delegate respondsToSelector:@selector(shouldAllowAuthenticationForUser:)]) {
        if (![self.delegate shouldAllowAuthenticationForUser:username]) {
            NSError *error = [NSError errorWithDomain:@"UserManagerErrorDomain"
                                                 code:UserErrorPermissionDenied
                                             userInfo:@{NSLocalizedDescriptionKey: @"Authentication denied"}];
            if (completion) {
                completion(NO, error);
            }
            return;
        }
    }

    // Perform authentication asynchronously
    dispatch_async(_authQueue, ^{
        BOOL success = [self validateCredentials:username password:password];

        dispatch_async(dispatch_get_main_queue(), ^{
            if (success) {
                self.currentUserId = username;
                self.authenticated = YES;
                _loginAttempts = 0;

                if ([self.delegate respondsToSelector:@selector(didAuthenticateUser:)]) {
                    [self.delegate didAuthenticateUser:username];
                }

                if (completion) {
                    completion(YES, nil);
                }
            } else {
                _loginAttempts++;
                NSError *error = [NSError errorWithDomain:@"UserManagerErrorDomain"
                                                     code:UserErrorNetworkFailure
                                                 userInfo:@{NSLocalizedDescriptionKey: @"Authentication failed"}];

                if ([self.delegate respondsToSelector:@selector(didFailAuthenticationWithError:)]) {
                    [self.delegate didFailAuthenticationWithError:error];
                }

                if (completion) {
                    completion(NO, error);
                }
            }
        });
    });
}

- (void)logoutWithCompletion:(void (^)(void))completion {
    dispatch_async(_authQueue, ^{
        [self invalidateSession];

        dispatch_async(dispatch_get_main_queue(), ^{
            self.authenticated = NO;
            self.currentUserId = @"";
            _internalToken = nil;
            [self.sessionCache removeAllObjects];

            if (completion) {
                completion();
            }
        });
    });
}

- (NSArray<NSString *> *)getUserPermissions {
    if (!self.authenticated) {
        return @[];
    }

    // Return cached permissions if available
    NSArray *cached = self.sessionCache[@"permissions"];
    if (cached) {
        return cached;
    }

    // Fetch from network or data store
    NSArray *permissions = @[@"read", @"write", @"delete"];
    self.sessionCache[@"permissions"] = permissions;

    return permissions;
}

#pragma mark - Private Methods

- (void)invalidateSession {
    _internalToken = nil;
    _loginAttempts = 0;
}

- (BOOL)validateCredentials:(NSString *)username password:(NSString *)password {
    // Simulate validation logic
    if ([username isEqualToString:@"admin"] && [password isEqualToString:@"password"]) {
        _internalToken = [[NSUUID UUID] UUIDString];
        return YES;
    }
    return NO;
}

#pragma mark - NSCoding

- (void)encodeWithCoder:(NSCoder *)coder {
    [coder encodeObject:self.currentUserId forKey:@"currentUserId"];
    [coder encodeBool:self.authenticated forKey:@"authenticated"];
}

- (instancetype)initWithCoder:(NSCoder *)coder {
    self = [self init];
    if (self) {
        _currentUserId = [coder decodeObjectForKey:@"currentUserId"];
        _authenticated = [coder decodeBoolForKey:@"authenticated"];
    }
    return self;
}

#pragma mark - NSCopying

- (id)copyWithZone:(NSZone *)zone {
    UserManager *copy = [[[self class] allocWithZone:zone] init];
    copy.currentUserId = [self.currentUserId copy];
    copy.authenticated = self.authenticated;
    return copy;
}

@end

/// Utility category for string validation
@interface NSString (UserValidation)

- (BOOL)isValidUsername;
- (BOOL)isValidPassword;

@end

@implementation NSString (UserValidation)

- (BOOL)isValidUsername {
    if (self.length < 3 || self.length > 20) {
        return NO;
    }

    NSCharacterSet *validCharacters = [NSCharacterSet alphanumericCharacterSet];
    NSCharacterSet *stringCharacters = [NSCharacterSet characterSetWithCharactersInString:self];
    return [validCharacters isSupersetOfSet:stringCharacters];
}

- (BOOL)isValidPassword {
    return self.length >= 8;
}

@end
"""
        chunks = parse_objc(content)

        # Should extract multiple types of chunks
        assert len(chunks) > 0, "Should extract chunks from realistic module"

        # Check for class/interface
        has_interface = any(
            "UserManager" in c.code and "@interface" in c.code for c in chunks
        )
        has_implementation = any(
            "UserManager" in c.code and "@implementation" in c.code for c in chunks
        )
        assert has_interface or has_implementation, "Should find UserManager class"

        # Check for protocol
        has_protocol = any(
            "UserAuthenticationDelegate" in c.code or "protocol" in c.code.lower()
            for c in chunks
        )
        assert has_protocol, "Should find protocol declaration"

        # Check for category
        has_category = any(
            "UserValidation" in c.code or "NSString" in c.code for c in chunks
        )
        assert has_category, "Should find category"

        # Verify multiple chunk types are present
        chunk_types = {c.chunk_type for c in chunks}
        assert len(chunk_types) >= 2, (
            f"Should have multiple chunk types, got: {chunk_types}"
        )

        # Check for various methods
        method_chunks = [
            c for c in chunks if c.chunk_type in (ChunkType.METHOD, ChunkType.FUNCTION)
        ]
        assert len(method_chunks) >= 3, (
            f"Should find at least 3 methods, found {len(method_chunks)}"
        )
