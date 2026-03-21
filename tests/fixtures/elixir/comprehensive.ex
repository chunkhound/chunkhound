defmodule MyApp.Accounts.User do
  @moduledoc """
  User context module for the accounts bounded context.
  Handles user CRUD operations and authentication.
  """

  use Ecto.Schema
  import Ecto.Changeset
  alias MyApp.Repo
  require Logger

  @type t :: %__MODULE__{
          name: String.t(),
          email: String.t()
        }

  @typep internal_state :: :active | :inactive

  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(user, attrs) do
    user
    |> cast(attrs, [:name, :email])
    |> validate_required([:name, :email])
    |> unique_constraint(:email)
  end

  @doc """
  Creates a new user with the given attributes.
  """
  def create_user(attrs) do
    %__MODULE__{}
    |> changeset(attrs)
    |> Repo.insert()
  end

  defp hash_password(password) do
    Bcrypt.hash_pwd_salt(password)
  end

  # Multi-clause function
  def status(:active), do: "Active"
  def status(:inactive), do: "Inactive"
  def status(_), do: "Unknown"

  defdelegate fetch(key), to: MyApp.Config

  defstruct [:name, :email, :age]
end

defprotocol Printable do
  @doc "Converts a value to a printable string"
  def to_string(value)
end

defimpl Printable, for: Integer do
  def to_string(value), do: Integer.to_string(value)
end

defmodule MyApp.GenServerExample do
  @moduledoc """
  A GenServer example with callbacks.
  """

  use GenServer

  @callback init(term()) :: {:ok, term()}

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(state) do
    {:ok, state}
  end

  @impl true
  def handle_call(:get, _from, state) do
    {:reply, state, state}
  end
end

defmodule MyApp.Guards do
  defguard is_positive(x) when is_integer(x) and x > 0
  defguardp is_valid_age(age) when is_integer(age) and age >= 0 and age <= 150

  defmacro debug(expr) do
    quote do
      IO.inspect(unquote(expr), label: "debug")
    end
  end

  defmacrop internal_macro(x) do
    quote do
      unquote(x) * 2
    end
  end
end
