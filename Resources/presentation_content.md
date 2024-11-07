# SeeClick Presentation Content

## Technical Overview Speech

Today I'm excited to present SeeClick, a groundbreaking advancement in visual GUI agents that fundamentally transforms how transformer-based models interact with graphical interfaces. This work from Nanjing University and Shanghai AI Lab introduces a novel architecture that bridges the gap between vision-language models and practical GUI automation.

Let me start with the core technical problem. Current GUI automation relies heavily on Large Language Models like GPT-4 processing structured DOM or HTML data. While this works, it faces three critical technical bottlenecks:
1. Structured data accessibility becomes impossible for native applications - imagine trying to get HTML from an iOS app
2. The context window gets overwhelmed - a simple webpage's HTML can consume thousands of tokens in the transformer's attention mechanism
3. Platform-specific observation spaces fragment our solutions - we need different architectures for HTML, DOM, and Android View Hierarchy

SeeClick's innovation lies in its transformer-based architecture that completely reimagines this approach. Instead of text-based representations, it leverages a Vision Transformer (ViT) encoder coupled with a Large Vision-Language Model through a specialized adapter layer. The key technical breakthrough here is what the authors call 'GUI grounding' - training the model to map natural language instructions directly to precise (x,y) coordinates or bounding boxes in interface screenshots.

The architecture comprises three key components:
1. Vision Encoder: A ViT (Vision Transformer) that processes screenshot inputs
2. Vision-Language Adapter: Connects the visual features to the language model
3. Large Language Model: Handles instruction understanding and action generation

Let's dive deeper into how these components work together. The backbone uses a modified vision encoder based on ViT, processing screenshots at 448x448 resolution. This feeds into a vision-language adapter that creates a unified representation space, allowing the Large Language Model to reason about visual elements and generate actions. The authors implemented this using LoRA for efficient fine-tuning of both the visual encoder and language model components.

The training pipeline is particularly sophisticated:
- GUI grounding pre-training utilizes:
 * 300K web pages with automatically extracted text-location pairs using a novel DOM parsing technique
 * Mobile UI data incorporating both widget captioning and hierarchical UI summarization
 * General vision-language instruction data to maintain LVLM capabilities
- They employ continual pre-training with action space modeling for:
 * Click operations as normalized (x,y) coordinates
 * Text input generation
 * Navigation command prediction

For benchmarking, they introduced ScreenSpot - a meticulously curated dataset spanning mobile, desktop, and web interfaces. What's technically interesting is their evaluation methodology across different element types: text, icons, widgets, and complex nested interfaces.

The results are remarkable from a transformer architecture perspective:
- On MiniWob: 73.6% success rate with just 2.8K training examples compared to the previous SOTA's 64.6% using 1.3M examples - that's two orders of magnitude more data efficient
- On ScreenSpot: 53.4% average accuracy across platforms, significantly outperforming GPT-4V's 16.2% in GUI grounding
- Most importantly, they demonstrated a direct correlation between enhanced GUI grounding capability and downstream task performance

This architecture essentially creates a foundation model for GUI interaction, achieving something unprecedented - a single transformer-based model that can interact with any graphical interface using pure visual input, much like human visual processing patterns.

## Audience Interaction Questions

### Question 1
"Looking at SeeClick's architecture, why do you think the authors chose to predict (x,y) coordinates directly as natural language outputs rather than using a more traditional classification approach with discrete position tokens like some other vision-language models?"

Discussion Points:
- Natural language coordinate prediction offers more flexibility
- No need for additional tokenization
- More precise localization
- Aligns with human intuition
- Coordinates can be generated as part of the natural language output
- Allows for continuous rather than discrete position values
- Integrates seamlessly with the model's existing language capabilities

### Question 2
"Given that SeeClick achieves better performance with just 0.3% of the training data compared to previous approaches, what do you think is the key factor enabling this data efficiency: the GUI grounding pre-training strategy or the unified vision-language architecture, and why?"

Discussion Points:
- GUI grounding pre-training provides specialized knowledge
- Unified architecture enables better transfer learning
- Quality of the pre-training data
- Effective leveraging of both visual and language capabilities
- Role of LoRA in efficient fine-tuning
- Importance of the vision-language adapter design
- Connection between pre-training objectives and downstream tasks


## Formal Pseudocode Description

```pseudocode
# SeeClick Model Architecture and Training Pipeline

# Main Model Components
class SeeClick:
    def __init__(self):
        self.vision_encoder = ViT()  # Vision Transformer for screenshot processing
        self.vision_lang_adapter = Adapter()  # Connects visual features to LLM
        self.llm = LargeLanguageModel()  # For instruction understanding & action generation
        
    def forward(self, screenshot, instruction, previous_actions):
        # Process screenshot through Vision Transformer
        visual_features = self.vision_encoder(screenshot)  # 448x448 resolution input
        
        # Connect visual and language features
        multimodal_features = self.vision_lang_adapter(visual_features)
        
        # Generate action based on instruction and visual context
        action = self.llm(instruction, multimodal_features, previous_actions)
        return action

# GUI Grounding Training
def train_gui_grounding(model, data):
    """
    Data includes:
    - Web UI data (300k pages)
    - Mobile UI data
    - General vision-language data
    """
    for screenshot, instruction, target_location in data:
        # Forward pass
        predicted_location = model(screenshot, instruction)
        
        # Location can be either:
        # 1. Point coordinates (x,y)
        # 2. Bounding box (left, top, right, bottom)
        loss = compute_location_loss(predicted_location, target_location)
        
        # Update using LoRA for efficient fine-tuning
        update_model_weights(model, loss)

# Action Space Definition
class ActionSpace:
    def define_action(self):
        return {
            'click': {'type': 4, 'coords': '(x,y)'},  # Click at coordinates
            'type': {'type': 3, 'text': 'string'},    # Type text
            'select': {'type': 2, 'value': 'string'}, # Select from dropdown
            'swipe': {'type': [0,1,8,9], 'direction': ['up','down','left','right']},
            'back': {'type': 5},
            'home': {'type': 6},
            'enter': {'type': 7}
        }

# Inference Pipeline
def inference(model, screenshot, instruction, previous_actions):
    # Format prompt
    prompt = f"""
    Please generate the next move according to the UI screenshot, 
    instruction and previous actions.
    Instruction: {instruction}
    Previous actions: {previous_actions}
    """
    
    # Get model prediction
    action = model(screenshot, prompt)
    
    # Action format example for click:
    # {"action_type": 4, "click_point": (0.49, 0.40)}
    return action
 ```   

## Implementation Demonstration

[Code walkthrough of the actual implementation also located in the Seeclick demo ipynb]

```python
# Model Initialization
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("cckevinn/SeeClick", 
                                           device_map="cuda",
                                           trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(
   "Qwen/Qwen-VL-Chat", 
   trust_remote_code=True
)

# Coordinate Prediction
def predict_location(image_path, instruction):
   prompt = f"In this UI screenshot, what is the position of the element corresponding to the command \"{instruction}\" (with point)?"
   query = tokenizer.from_list_format([
       {'image': image_path},
       {'text': prompt}
   ])
   response, _ = model.chat(tokenizer, query=query, history=None)
   return response

# Visualization
def display_images(img_path, point, text):
   img = Image.open(img_path)
   img_array = np.array(img)
   
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
   
   ax1.imshow(img_array)
   ax1.set_title("Original Image")
   ax1.axis('off')
   
   ax2.imshow(img_array)
   x, y = point
   x_px = x * img_width
   y_px = y * img_height
   ax2.plot(x_px, y_px, 'ro', markersize=10, markeredgecolor='white')
   ax2.set_title(text)
   ax2.axis('off')
   
   plt.tight_layout()
   plt.show()
   ```

## Critical Analysis

## Technical Limitations:

- Limited action space (basic clicking and typing only)
- Performance dependency on open-source LVLMs
- Resolution constraints with high-resolution screenshots


### What Could Have Been Developed Further:

- Cross-platform optimization
- Error recovery mechanisms
- Complex interaction sequences
- Better handling of dynamic interfaces


### Important Aspects Not Fully Addressed:

- Latency and real-time performance
- Security implications
- Robustness to interface changes
- Handling of overlapping elements


### Comparison Limitations:

- Model size differences in evaluations
- Limited commercial solution comparisons
- Success metrics might not reflect real-world usage
- Need for more diverse test scenarios


### Potential Improvements:

- Enhanced grounding for non-text elements
- Better handling of long-range dependencies
- More sophisticated action space modeling
- Improved resolution handling for web interfaces


## Conclusion

SeeClick represents a significant advancement in visual GUI agents through:

- Novel GUI grounding approach
- Efficient training methodology
- Universal applicability across platforms
- Strong performance with minimal training data

### Future research directions include:

- Expanding the action space
- Improving resolution handling
- Enhancing cross-platform capabilities
- Developing more robust error recovery mechanisms

The combination of GUI grounding pre-training and unified architecture opens new possibilities for more intuitive and efficient GUI automation.
