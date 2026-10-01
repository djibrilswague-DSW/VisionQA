# VisionQA

## Vision Assistant multimodal — VLM & Computer Vision

VisionQA est une application de question-réponse visuelle basée sur un modèle de vision-langage (VLM).

L'utilisateur fournit une image et pose une question en langage naturel. Le modèle analyse l'image et retourne une réponse structurée contenant une description de l'image, les objets identifiés et la réponse à la question.

## Fonctionnalités

- Analyse d'images avec un modèle Vision-Language
- Question-réponse visuelle (VQA)
- Réponses structurées au format JSON
- Identification d'objets présents dans l'image
- Interface utilisateur avec Gradio
- Organisation modulaire du projet
- Support Docker pour la reproductibilité

## Modèle

Le projet utilise :

- **Qwen2.5-VL-3B-Instruct**
- Hugging Face Transformers
- PyTorch

Le modèle est capable de traiter simultanément des informations visuelles et textuelles afin de répondre à des questions portant sur une image.

## Architecture

```text
VisionQA/
│
├── app/
│   ├── model.py        # Chargement du modèle et du processor
│   ├── visionqa.py     # Pipeline d'analyse image + question
│   └── main.py         # Interface Gradio
│
├── examples/           # Exemples d'utilisation
│
├── tests/
│   └── test_visionqa.py
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md