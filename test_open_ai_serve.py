import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="token-abc123",
)

# completion = client.chat.completions.create(
#     model="./pretrained_models/SmolVLM-135M",
#     messages=[
#         {"role": "user", "content": "Hello! Who is America's president?"},
#     ],
# )

# print(completion.choices[0].message.content)


# ── Option A: image from local file ───────────────────────────────
def image_to_base64(path: str | Path) -> str:
    path = Path(path)
    base64_image = base64.b64encode(path.read_bytes()).decode("utf-8")
    # change jpeg to png if needed
    return f"data:image/jpeg;base64,{base64_image}"


image_path = "images/ava.webp"
image_data_url = image_to_base64(image_path)



response = client.chat.completions.create(
    model="./pretrained_models/SmolVLM-135M",          # or "gpt-4o", "gpt-4-turbo"
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail and tell me what programming concept it is illustrating."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url,
                        # optional → lower quality = cheaper & faster
                        # "detail": "low"     # or "high" (default=auto)
                    }
                }
            ]
        }
    ],
    max_tokens=400,
    temperature=0.9
)

print(response.choices[0].message.content)
