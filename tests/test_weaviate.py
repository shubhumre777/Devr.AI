import pytest
from backend.app.database.weaviate.client import get_client
from uuid import uuid4
from backend.app.models.database.weaviate import (
    WeaviateRepository,
    WeaviatePullRequest,
    WeaviateUserProfile,
)

@pytest.mark.asyncio
async def test_weaviate_client():
    client = get_client()
    await client.connect()
    try:
        assert client is not None, "Weaviate client should not be None"
        ready = await client.is_ready()
        assert ready, "Weaviate client is not ready"
    except Exception as e:
        raise AssertionError(f"Weaviate client connection failed: {e}")
    finally:
        await client.close()

async def insert_user_profile(client):
    user_profile = WeaviateUserProfile(
        user_id=str(uuid4()),
        github_username="testuser",
        profile_text_for_embedding="This is a test profile"
    )
    try:
        questions = client.collections.get("weaviate_user_profile")
        # Exclude array properties to match TEXT schema and explicitly set the Weaviate UUID
        props = user_profile.dict(by_alias=True, exclude={'repositories', 'pull_requests'})
        await questions.data.insert(
            properties=props,
            uuid=user_profile.user_id
        )
        print("User profile inserted successfully.")
        return user_profile
    except Exception as e:
        print(f"Error inserting user profile: {e}")
        return None

async def get_user_profile_by_id(client, user_id: str):
    try:
        questions = client.collections.get("weaviate_user_profile")
        # Exact lookup using the UUID instead of BM25 text search
        response = await questions.query.fetch_object_by_id(user_id)
        if response:
            return WeaviateUserProfile(**response.properties)
    except Exception as e:
        print(f"Error retrieving user profile: {e}")
        return None

async def delete_user_profile(client, user_id: str):
    try:
        questions = client.collections.get("weaviate_user_profile")
        deleted = await questions.data.delete_by_id(user_id)
        if deleted:
            print("User profile deleted successfully.")
            return True
        else:
            print("User profile not found.")
            return False
    except Exception as e:
        print(f"Error deleting user profile: {e}")
        return False

@pytest.mark.asyncio
async def test_user_profile():
    client = get_client()
    await client.connect()
    try:
        inserted_user = await insert_user_profile(client)
        assert inserted_user is not None, "User profile insertion failed"
        
        retrieved_user = await get_user_profile_by_id(client, inserted_user.user_id)
        assert retrieved_user is not None, "Failed to retrieve the inserted user"
        assert str(retrieved_user.user_id) == str(inserted_user.user_id), "Retrieved ID does not match"
        
        deleted = await delete_user_profile(client, inserted_user.user_id)
        assert deleted is True, "Failed to delete the user profile"
    finally:
        await client.close()