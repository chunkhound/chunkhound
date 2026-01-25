-- Sample Lua file for parser testing
-- This file contains various Lua constructs to test the parser

-- Global function
function greet(name)
    print("Hello, " .. name)
end

-- Local function
local function calculateSum(a, b)
    return a + b
end

-- Function with multiple returns
local function divmod(a, b)
    return math.floor(a / b), a % b
end

-- Table as a module/class pattern
local MyModule = {}

function MyModule.init(self, value)
    self.value = value
end

function MyModule:getValue()
    return self.value
end

-- Method syntax (colon)
function MyModule:setValue(value)
    self.value = value
end

-- Local variables
local config = {
    debug = true,
    version = "1.0.0",
    maxRetries = 3
}

-- Require/import pattern
local json = require("cjson")
local utils = require("lib.utils")

-- Conditional blocks
local function processData(data)
    if data == nil then
        return nil
    elseif type(data) == "table" then
        return data
    else
        return {data}
    end
end

-- Loop constructs
local function iterateItems(items)
    -- Numeric for loop
    for i = 1, #items do
        print(items[i])
    end

    -- Generic for loop (pairs)
    for key, value in pairs(items) do
        print(key, value)
    end

    -- While loop
    local count = 0
    while count < 10 do
        count = count + 1
    end

    -- Repeat until
    repeat
        count = count - 1
    until count == 0
end

-- Anonymous function / closure
local callback = function(x)
    return x * 2
end

-- Higher-order function
local function map(tbl, fn)
    local result = {}
    for i, v in ipairs(tbl) do
        result[i] = fn(v)
    end
    return result
end

-- Metatables (OOP pattern)
local Animal = {}
Animal.__index = Animal

function Animal.new(name)
    local self = setmetatable({}, Animal)
    self.name = name
    return self
end

function Animal:speak()
    print(self.name .. " makes a sound")
end

-- Coroutine example
local function producer()
    return coroutine.create(function()
        for i = 1, 5 do
            coroutine.yield(i)
        end
    end)
end

--[[
    Multi-line comment block
    This is a documentation comment
    explaining the module purpose
]]

-- Error handling pattern
local function safeCall(fn, ...)
    local status, result = pcall(fn, ...)
    if not status then
        print("Error: " .. result)
        return nil
    end
    return result
end

-- Return module table
return MyModule
