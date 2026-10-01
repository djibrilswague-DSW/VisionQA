from PIL import Image
import json

from app.model import load_model


model, processor = load_model()


def analyze_image(image, question):

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.copy()
    image.thumbnail((768, 768))

    prompt = f"""
Analyse cette image et réponds à la question.

Retourne uniquement un objet JSON valide avec exactement ces quatre champs :

{{
    "description": "description courte de l'image",
    "objects": ["objet 1", "objet 2"],
    "question": "{question}",
    "answer": "réponse courte à la question"
}}

Question : {question}
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=150
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    output = output.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(output)
    except json.JSONDecodeError:
        result = {
            "description": "",
            "objects": [],
            "question": question,
            "answer": output
        }

    return result