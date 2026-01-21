# Rails model with DSL patterns
class Post < ApplicationRecord
  belongs_to :author, class_name: 'User'
  has_many :comments, dependent: :destroy
  has_many :tags, through: :post_tags

  validates :title, presence: true, length: { minimum: 5 }
  validates :email, uniqueness: true

  before_save :normalize_title
  after_create :send_notification

  scope :published, -> { where(published: true) }
  scope :recent, -> { order(created_at: :desc).limit(10) }

  def normalize_title
    self.title = title.downcase
  end

  private

  def send_notification
    # Send notification
  end
end
