# Basic Ruby file with various constructs
require 'json'
require_relative 'helper'

# Maximum retries constant
MAX_RETRIES = 3

# A sample module
module Utils
  def self.format_string(str)
    str.strip.downcase
  end

  def self.process(data)
    # Process the data
    data.map { |x| x * 2 }
  end
end

# A sample class
class User
  attr_accessor :name, :email

  def initialize(name, email)
    @name = name
    @email = email
  end

  # Instance method
  def greet
    "Hello, #{@name}!"
  end

  # Class method
  def self.create(name, email)
    new(name, email)
  end

  private

  def normalize_email
    @email.downcase
  end
end

# Subclass example
class AdminUser < User
  def initialize(name, email, role)
    super(name, email)
    @role = role
  end

  def admin?
    true
  end
end
