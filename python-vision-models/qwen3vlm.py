"""
Qwen3-0.6B Interactive Chat Demo

This script demonstrates how to have a conversation with the Qwen3-0.6B model.
It supports both single questions and multi-turn conversations.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_qwen_model():
    """Load the Qwen3-0.6B model and tokenizer"""
    print("Loading Qwen3-0.6B model...")
    print("This may take a moment to download and load the model...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ Model loaded successfully!")
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def generate_response(tokenizer, model, messages, max_new_tokens=100):
    """Generate a response from the model"""
    try:
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
        
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return None

def single_question_mode(tokenizer, model):
    """Single question mode - ask one question and get an answer"""
    print("\n" + "="*50)
    print("Single Question Mode")
    print("="*50)
    
    question = input("Enter your question: ").strip()
    if not question:
        question = "Who are you?"
        print(f"Using default question: {question}")
    
    messages = [{"role": "user", "content": question}]
    
    print(f"\n🤖 Qwen3: Thinking...")
    response = generate_response(tokenizer, model, messages)
    
    if response:
        print(f"🤖 Qwen3: {response}")
    else:
        print("❌ Failed to generate response")

def conversation_mode(tokenizer, model):
    """Multi-turn conversation mode"""
    print("\n" + "="*50)
    print("Conversation Mode")
    print("="*50)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    
    conversation_history = []
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'clear':
            conversation_history = []
            print("🧹 Conversation cleared!")
            continue
        elif not user_input:
            continue
        
        # Add user message to conversation
        conversation_history.append({"role": "user", "content": user_input})
        
        print("🤖 Qwen3: Thinking...")
        response = generate_response(tokenizer, model, conversation_history)
        
        if response:
            print(f"🤖 Qwen3: {response}")
            # Add assistant response to conversation
            conversation_history.append({"role": "assistant", "content": response})
        else:
            print("❌ Failed to generate response")

def demo_questions(tokenizer, model):
    """Run a demo with predefined questions"""
    print("\n" + "="*50)
    print("Demo Questions")
    print("="*50)
    
    demo_questions = [
        "Who are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "Write a short poem about AI",
        "What are the benefits of renewable energy?"
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n--- Question {i}: {question} ---")
        
        messages = [{"role": "user", "content": question}]
        response = generate_response(tokenizer, model, messages)
        
        if response:
            print(f"🤖 Qwen3: {response}")
        else:
            print("❌ Failed to generate response")
        
        print("-" * 40)

def main():
    print("🤖 Qwen3-0.6B Interactive Chat Demo")
    print("=" * 50)
    
    # Load model
    tokenizer, model = load_qwen_model()
    
    if tokenizer is None or model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    print("\nChoose a mode:")
    print("1. Single Question")
    print("2. Conversation Mode")
    print("3. Demo Questions")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        single_question_mode(tokenizer, model)
    elif choice == "2":
        conversation_mode(tokenizer, model)
    elif choice == "3":
        demo_questions(tokenizer, model)
    else:
        print("Invalid choice. Running single question mode...")
        single_question_mode(tokenizer, model)

if __name__ == "__main__":
    main()