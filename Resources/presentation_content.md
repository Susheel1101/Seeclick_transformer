# SeeClick Presentation Content

## Technical Overview

# SeeClick: A Visual GUI Agent

## Core Technical Problem
Current GUI automation faces three critical technical bottlenecks:

1. **Structured Data Accessibility**
   - Impossible for native applications
   - Example: Cannot extract HTML from iOS apps

2. **Context Window Limitations**
   - Simple webpage HTML consumes thousands of tokens
   - Overloads transformer attention mechanism

3. **Platform Fragmentation**
   - Different architectures needed for:
     * HTML
     * DOM
     * Android View Hierarchy

## Technical Innovation
SeeClick introduces a transformer-based architecture with:
- Vision Transformer (ViT) encoder
- Large Vision-Language Model
- Specialized adapter layer
- Key breakthrough: 'GUI grounding'

### Key Components
1. Vision Encoder
   - ViT for screenshot processing
   - 448x448 resolution input

2. Vision-Language Adapter
   - Connects visual features to language model
   - Creates unified representation space

3. Large Language Model
   - Instruction understanding
   - Action generation

### Training Pipeline
**GUI Grounding Pre-training:**
- 300K web pages with text-location pairs
- Mobile UI data:
  * Widget captioning
  * Hierarchical UI summarization
- General vision-language instruction data

**Action Space Modeling:**
- Click operations (normalized x,y coordinates)
- Text input generation
- Navigation command prediction

## Evaluation & Results

### Benchmarks
1. **MiniWob**
   - 73.6% success rate with 2.8K examples
   - Previous SOTA: 64.6% with 1.3M examples
   - Two orders of magnitude more data efficient

2. **ScreenSpot**
   - 53.4% average accuracy across platforms
   - Outperforms GPT-4V (16.2%) in GUI grounding
   - Evaluation across:
     * Text elements
     * Icons
     * Widgets
     * Complex nested interfaces

## Achievement
Created a foundation model for GUI interaction that:
- Uses pure visual input
- Works across any graphical interface
- Mirrors human visual processing patterns
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
