import gradio as gr

from app.visionqa import analyze_image


def run_visionqa(image, question):

    if image is None:
        return "Aucune image fournie.", "", ""

    if not question.strip():
        return "Veuillez entrer une question.", "", ""

    result = analyze_image(image, question)

    description = result.get("description", "")
    objects = ", ".join(result.get("objects", []))
    answer = result.get("answer", "")

    return description, objects, answer


demo = gr.Interface(
    fn=run_visionqa,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Textbox(
            label="Question",
            placeholder="Posez une question sur l'image..."
        )
    ],
    outputs=[
        gr.Textbox(label="Description"),
        gr.Textbox(label="Objets détectés"),
        gr.Textbox(label="Réponse")
    ],
    title="VisionQA",
    description="Assistant multimodal basé sur Qwen2.5-VL."
)


if __name__ == "__main__":
    demo.launch()