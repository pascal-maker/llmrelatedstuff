import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedVQA:
    def __init__(self):
        self.model_name = "Qwen/Qwen2-VL-2B-Instruct"  # Use instruct version for better performance
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """Load model with optimized settings"""
        try:
            logger.info(f"Loading model on {self.device}...")
            
            # Load model with optimizations
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                attn_implementation="flash_attention_2" if self.device == "cuda" else "eager"
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to basic pipeline
            from transformers import pipeline
            self.pipeline = pipeline(
                task="image-text-to-text",
                model=self.model_name,
                device_map="auto",
                torch_dtype="auto"
            )
    
    def preprocess_image(self, image):
        """Enhanced image preprocessing"""
        if image is None:
            return None
            
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (maintain aspect ratio)
        max_size = 1024
        width, height = image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def enhance_question(self, question):
        """Add context to improve question understanding"""
        question = question.strip()
        
        # Add helpful prefixes for better responses
        if question.lower().startswith(('what', 'who', 'where', 'when', 'why', 'how')):
            return f"Please analyze this image carefully and answer: {question}"
        else:
            return f"Looking at this image, {question}"
    
    def answer_question(self, image, question):
        """Enhanced VQA with better error handling and processing"""
        
        # Input validation
        if image is None:
            return "❌ Please upload an image first."
        
        if not question or question.strip() == "":
            return "❌ Please ask a question about the image."
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            enhanced_question = self.enhance_question(question)
            
            # Use direct model if available, otherwise fallback to pipeline
            if hasattr(self, 'model') and self.model is not None:
                return self._answer_with_model(processed_image, enhanced_question)
            else:
                return self._answer_with_pipeline(processed_image, enhanced_question)
                
        except Exception as e:
            logger.error(f"Error in VQA: {e}")
            return f"❌ Error processing your request: {str(e)}"
    
    def _answer_with_model(self, image, question):
        """Answer using direct model access for better control"""
        try:
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            return f"❌ Error during model inference: {str(e)}"
    
    def _answer_with_pipeline(self, image, question):
        """Fallback to pipeline method"""
        try:
            result = self.pipeline({"image": image, "text": question})
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No response generated.")
            elif isinstance(result, dict):
                return result.get("generated_text", "No response generated.")
            else:
                return str(result)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return f"❌ Pipeline error: {str(e)}"

# Initialize the VQA system
vqa_system = EnhancedVQA()

# Enhanced Gradio interface
def create_interface():
    with gr.Blocks(title="Enhanced Visual Q&A", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 Enhanced Visual Q&A with Qwen2-VL
        
        Upload an image and ask detailed questions about it. This enhanced version includes:
        - Better image preprocessing
        - Improved question understanding
        - Enhanced error handling
        - Optimized model loading
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil", 
                    label="📷 Upload Image",
                    height=400
                )
                question_input = gr.Textbox(
                    lines=3,
                    label="❓ Ask a Question",
                    placeholder="Examples:\n• What objects do you see in this image?\n• Describe the scene in detail\n• What is the person doing?\n• What colors are prominent?",
                    info="Be specific for better results!"
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Analyze", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            with gr.Column():
                output = gr.Textbox(
                    label="📝 Analysis Result",
                    lines=10,
                    max_lines=20,
                    show_copy_button=True
                )
        
        # Example questions
        gr.Markdown("### 💡 Example Questions:")
        example_questions = [
            "What objects can you identify in this image?",
            "Describe the scene and setting in detail",
            "What is the main subject doing?",
            "What are the dominant colors in this image?",
            "Can you read any text in this image?",
            "What is the mood or atmosphere of this scene?"
        ]
        
        for i, example in enumerate(example_questions):
            gr.Button(example, size="sm").click(
                lambda x=example: x, outputs=question_input
            )
        
        # Event handlers
        submit_btn.click(
            fn=vqa_system.answer_question,
            inputs=[image_input, question_input],
            outputs=output
        )
        
        clear_btn.click(
            lambda: (None, "", ""),
            outputs=[image_input, question_input, output]
        )
        
        # Auto-submit on Enter
        question_input.submit(
            fn=vqa_system.answer_question,
            inputs=[image_input, question_input],
            outputs=output
        )
    
    return demo

# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,  # Set to True for public sharing
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        show_error=True
    )