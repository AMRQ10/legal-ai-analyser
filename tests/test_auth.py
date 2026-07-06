def test_register_new_user(client, test_user_data):
    response = client.post("/auth/register", json=test_user_data)
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == test_user_data["email"]
    assert "hashed_password" not in data  # password must never be exposed

def test_register_duplicate_email_fails(client, test_user_data):
    client.post("/auth/register", json=test_user_data)
    response = client.post("auth/register", json=test_user_data)
    assert response.status_code == 400

def test_login_with_correct_credentials(client, test_user_data):
    client.post("/auth/register", json=test_user_data)
    response = client.post("/auth/login", json={
        "email": test_user_data["email"],
        "password": test_user_data["password"]
    })
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_login_with_wrong_password_fails(client, test_user_data):
    client.post("/auth/register", json=test_user_data)
    response = client.post("/auth/login", json={
        "email": test_user_data["email"],
        "password": "WrongPassword"
    })
    assert response.status_code == 401

def test_login_with_nonexistent_email_fails(client):
    response = client.post("/auth/login", json={
        "email": "doesnotexist@example.com",
        "password": "anything"
    })
    assert response.status_code == 401

def test_protected_endpoint_with_valid_token_succeeds(authenticated_client):
    response = authenticated_client.get("/auth/me")
    assert response.status_code == 200

def test_get_me_returns_correct_user(authenticated_client, test_user_data):
    response = authenticated_client.get("/auth/me")
    data = response.json()
    assert data["email"] == test_user_data["email"]


