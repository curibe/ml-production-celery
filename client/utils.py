import httpx


async def generate_images(api_endpoint, params):
    async with httpx.AsyncClient() as client:
        response = await client.post(api_endpoint, json=params, timeout=120)
    return response


def get_dimensions(size):
    size_dict = {
        "512x512": (512, 512),
        "768x768": (768, 768)
    }
    if size in size_dict:
        return size_dict[size]
    else:
        raise ValueError(f"Invalid size: {size}")