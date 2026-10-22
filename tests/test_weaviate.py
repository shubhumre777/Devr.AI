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
    assert client is not None, "Weaviate client should not be None"
    try:
        ready = await client.is_ready()
        assert ready, "Weaviate client is not ready"
    except Exception as e:
        raise AssertionError(f"Weaviate client connection failed: {e}")

async def insert_user_profile():
    user_profile = WeaviateUserProfile(
        user_id=str(uuid4()),
        github_username="testuser",
        profile_text_for_embedding="This is a test profile"
    )
    client = get_client()
    try:
        questions = client.collections.get("weaviate_user_profile")
        await questions.data.insert(properties=user_profile.dict(by_alias=True))
        print("User profile inserted successfully.")
        return user_profile
    except Exception as e:
        print(f"Error inserting user profile: {e}")
        return None

async def get_user_profile_by_id(user_id: str):
    client = get_client()
    try:
        questions = client.collections.get("weaviate_user_profile")
        response = await questions.query.bm25(
            query=user_id,
            return_properties=["userId", "githubUsername", "profileTextForEmbedding"]
        )
        if response and len(response.objects) > 0:
            user_profile_data = response.objects[0].properties
            return WeaviateUserProfile(**user_profile_data)
    except Exception as e:
        print(f"Error retrieving user profile: {e}")
        return None

async def delete_user_profile(user_id: str):
    client = get_client()
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
    inserted_user = await insert_user_profile()
    assert inserted_user is not None, "User profile insertion failed"
    await get_user_profile_by_id(inserted_user.user_id)
    await delete_user_profile(inserted_user.user_id)