import asyncio

import httpx

from config import size_dict


async def generate_images(api_endpoint, params):
    async with httpx.AsyncClient() as client:
        response = await client.post(api_endpoint, json=params, timeout=120)
    return response


async def long_poll_task_result(task_id, result_url, max_retries=20, delay=5):
    async with httpx.AsyncClient() as client:
        for _ in range(max_retries):
            response = await client.get(f"{result_url}/{task_id}")

            if response.status_code == 200:
                # The task is finished
                return response.content
            elif response.status_code == 202:
                # The task is not finished yet
                print(response.json())
            else:
                # Error occurred
                print(f"Error: {response.status_code}")
                return None

        # Wait before next retry
        await asyncio.sleep(delay)

        # Reached maximum retries
        return None


def get_dimensions(size):
    if size in size_dict:
        return size_dict[size]
    else:
        raise ValueError(f"Invalid size: {size}")
