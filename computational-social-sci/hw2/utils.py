from ipywidgets import (
    Button,
    HTML,
    HBox,
    VBox,
    ToggleButtons
)
import os
from ipywidgets import Button, HTML, HBox
from IPython.display import display
import random
import pickle
import time
import pandas as pd
from copy import deepcopy


def load_samples(data_path):
    path_samples = data_path
    assert os.path.exists(path_samples), f"{path_samples} does not exist"
    samples = pd.read_csv(path_samples, header=0)
    return samples.to_dict("records")


def save_annotations(path, annots, verbose=False):
    with open(path, "wb") as file:
        pickle.dump(annots, file)
    if verbose:
        print(f"{len(annots)} annotation saved to {path}")


def load_annotations(save_path, examples):
    if os.path.isfile(save_path):
        with open(save_path, "rb") as file:
            loaded_annotations = pickle.load(file)
        annotated = [annot["talkid"] for annot in loaded_annotations['annots']]
        examples_out = [example for example in examples if example['talkid'] not in annotated]
        return loaded_annotations, examples_out
    else:
        return {"demographics": {}, "annots": []}, examples
    
def export_csv(annotations_path, save_path):
    if os.path.exists(annotations_path):
        with open(annotations_path, "rb") as file:
            loaded_annotations = pickle.load(file)
        exporting = []
        for annot in loaded_annotations['annots']:
            example = {
                "gender": annot['gender'],
                "comment": annot['comment'],
                "talkid": annot['talkid'],
                "videourl": annot['videourl']
            }
            if "annotation" in annot:
                annotations = annot['annotation']
                example.update(annotations)
            exporting.append(example)
        pd.DataFrame(exporting).to_csv(save_path, index=None)
    else:
        raise FileExistsError("Annotations file does not exists")


def display_buttons(btns, num_buttons_in_a_row=3):
    if len(btns) > num_buttons_in_a_row:
        box = VBox([HBox(btns[x:x + num_buttons_in_a_row])
                    for x in range(0, len(btns), num_buttons_in_a_row)])
    else:
        box = HBox(btns)
    display(box)


def get_demographics():
    button_sets = {}        
    
    button_sets['gender'] = ToggleButtons(
        options=["Female", "Male"],
        description='You personally identify as:',
        disabled=False,
        value=None,
        button_style='',
    )
    
    button_sets['age'] = ToggleButtons(
        options=["0-15", "15-25", "25-35", "35+" ],
        description='Please select the correct age bracket:',
        disabled=False,
        value=None,
        button_style='',
    )
    for set, buttons in button_sets.items():
        display(buttons)
    return button_sets


def save_demographics(save_path, demographics, verbose=False):
    demographics = {name: buttons.value for name, buttons in demographics.items()}
    if os.path.isfile(save_path):
        with open(save_path, "rb") as file:
            loaded_annotations = pickle.load(file)
            loaded_annotations['demographics'] = demographics
    else:
        loaded_annotations = {"demographics": demographics, "annots": []}
    with open(save_path, "wb") as file:
        pickle.dump(loaded_annotations, file)
    if verbose:
        print(f"demographics saved to {save_path}")


def annotate(examples, 
             save_path,
             questions=[
                 {"name": 'tone', 
                  "question": 'What is the tone of the response with regards to the original poster or speaker?', 
                  "options": ["Strongly Negative", "Negative", "Neutral", "Positive", "Strongly Positive", "None"]},
                 {"name": 'expertise', "question": 'Based on the comment, what do you think is the expertise of the speaker?', 
                  "options": ["STEM degree", "Non-STEM degree", "No Degree"]},
                 {"name": 'encouraging', "question": 'Overall, is this response:', "options": ["Encouraging", "Discouraging"]},
                 {"name": 'respectful', "question": 'Overall, is this response:', "options": ["Disrespectful", "Neutral", "Respectful"]}
                 ], 
             shuffle=False, 
             final_process_fn=None):
    
    def render(index):
        nonlocal start_time
        set_label_text(index)
        
        if index >= len(examples):
            end_time = time.time()
            print(f'Annotation done. (took {end_time-start_time:.0f}s)')
            if final_process_fn is not None:
                final_process_fn(list(annotations.items()))
            for _, butn in responses.items():
                butn.disabled = True
            btn.disabled = True
            count_label.value = f'{len(annotations["annots"])} of {len(annotations["annots"])} Examples annotated, Current Position: {len(annotations["annots"])} '
            return
        
        comment = examples[index]["comment"]
        parsed = comment.replace("\n", "<br>")
        text = '<h4 class="text-center"><u>Consider the following internet video, and its corresponding response:</u></h4>'
        text += f'<div class="row"><p class="text-center" style="padding-left:10%;padding-right:10%;font-size:120%"><br>{parsed}<br></p>  </div>'
        sentence_html.value = text
        for set, buttons in responses.items():
            buttons.value = None
            
    def set_label_text(index):
        nonlocal count_label
        if len(annotations["annots"])==0:
            count_label.value = f'{len(annotations["annots"])} of {len(examples)} Examples annotated, Current Position: {index + 1}'
        else:
            nonlocal start_time
            time_took = time.time() - start_time
            count_label.value = f'{len(annotations["annots"])} of {len(examples)} Examples annotated, Current Position: {index + 1}, Took {time_took:.0f}s (Avg {time_took/len(annotations):.0f}s) '
            
    def next_example(btn=None):
        nonlocal current_index
        if current_index < len(examples):
            current_index += 1
            render(current_index)
    
    def add_annotation(annotation):
        if "annotation" in examples[current_index]:
            del examples[current_index]["annotation"]
        examples[current_index]["annotation"] = annotation
        annotations["annots"].append(examples[current_index])
        save_annotations(save_path, annotations)
        next_example()
        
    if shuffle:
        random.shuffle(examples)
    annotations, examples = load_annotations(save_path, examples)
    current_index = 0
    
    start_time = time.time()
    count_label = HTML()
    set_label_text(current_index)
    display(count_label)

    #### SHOWING MAIN PART
    sentence_html=HTML()
    display(sentence_html)
    responses = {
        el['name']: ToggleButtons(
            options=el['options'], 
            description=el['question'], 
            disabled=False, value=None, button_style='') for el in questions
    }        

    
    for set, buttons in responses.items():
        display(buttons)
    
    btn = Button(description='submit', button_style='info', icon='check')
    def on_click(btn):
        nonlocal current_index
        labels_on = {}
        for cat, buttons in responses.items():
            labels_on[cat] = buttons.value
            buttons.value = None
        add_annotation(labels_on)
        render(current_index)
        
    btn.on_click(on_click)
    display(btn)
    render(current_index)
    return
    

def annotate_two(examples, 
             save_path,
             questions=[
                 {"name": 'tone', 
                  "question": """What is the tone of the response with regards to the original poster or speaker?
                  Here are some examples of what each option might look like:
                    Strongly Negative: "You are the biggest idiot I have ever. Why would you ever even think and believe these shenanigans, much less present it as a talk to this many people? Don't ever speak on the internet again!"
                    Negative: "You are wrong. This isn't correct, you should have more thoroughly researched your topic."
                    Neutral: "I disagree with the statement in the video. But there are many valuable points to consider from a different perspective that made me reflect on my beliefs."
                    Positive: "I think you are right. I have always thought the same way and I am glad to see someone else who thinks the same way! Thank you for giving the talk, it really helped put the words in my mouth that I couldn't say."
                    Strongly Positive: "You are the most intelligent person I have ever heard, know, come across, EVERYTHING. I am so glad that you are speaking to the world, and that I was born with ears so I would be able to listen to the glory of your words and make me see the world in a brand new light. This was the most impactful 10 minutes of my life, truly transformational. What you've accomplished in this talk will be forever remembered by me and all of my future generations! I hope you continue to speak and share your knowledge with the rest of the world, they absolutely need to hear what you preach!"
                  """, 
                  "options": ["Strongly Negative", "Negative", "Neutral", "Positive", "Strongly Positive", "None"]},
                 {"name": 'expertise', "question": 'Based on the comment, what do you think is the expertise of the speaker?', 
                  "options": ["No Degree", "Non-STEM degree", "STEM degree"]},
                 {"name": 'encouraging', "question": 'Overall, is this response:', "options": ["Discouraging", "Encouraging"]},
                 {"name": 'respectful', "question": 'Overall, is this response:', "options": ["Disrespectful", "Neutral", "Respectful"]}
                 ], 
             shuffle=False, 
             final_process_fn=None):
    
    def render(index):
        nonlocal start_time
        set_label_text(index)
        
        if index >= len(examples):
            end_time = time.time()
            print(f'Annotation done. (took {end_time-start_time:.0f}s)')
            if final_process_fn is not None:
                final_process_fn(list(annotations.items()))
            for _, butn in responses.items():
                butn.disabled = True
            btn.disabled = True
            count_label.value = f'{len(annotations["annots"])} of {len(annotations["annots"])} Examples annotated, Current Position: {len(annotations["annots"])} '
            return
        
        comment = examples[index]["comment"]
        parsed = comment.replace("\n", "<br>")
        text = '<h4 class="text-center"><u>Consider the following internet video, and its corresponding response:</u></h4>'
        text += f'<div class="row"><p class="text-center" style="padding-left:10%;padding-right:10%;font-size:120%"><br>{parsed}<br></p>  </div>'
        sentence_html.value = text
        for set, buttons in responses.items():
            buttons.value = None
            
    def set_label_text(index):
        nonlocal count_label
        if len(annotations["annots"])==0:
            count_label.value = f'{len(annotations["annots"])} of {len(examples)} Examples annotated, Current Position: {index + 1}'
        else:
            nonlocal start_time
            time_took = time.time() - start_time
            count_label.value = f'{len(annotations["annots"])} of {len(examples)} Examples annotated, Current Position: {index + 1}, Took {time_took:.0f}s (Avg {time_took/len(annotations):.0f}s) '
            
    def next_example(btn=None):
        nonlocal current_index
        if current_index < len(examples):
            current_index += 1
            render(current_index)
    
    def add_annotation(annotation):
        if "annotation" in examples[current_index]:
            del examples[current_index]["annotation"]
        examples[current_index]["annotation"] = annotation
        annotations["annots"].append(examples[current_index])
        save_annotations(save_path, annotations)
        next_example()
        
    if shuffle:
        random.shuffle(examples)
    annotations, examples = load_annotations(save_path, examples)
    current_index = 0
    
    start_time = time.time()
    count_label = HTML()
    set_label_text(current_index)
    display(count_label)

    #### SHOWING MAIN PART
    sentence_html=HTML()
    display(sentence_html)
    """
    responses = {
        el['name']: ToggleButtons(
            options=el['options'], 
            description=el['question'], 
            disabled=False, value=None, button_style='') for el in questions
    }        

    
    for set, buttons in responses.items():
        display(buttons)
    """

    """
    # Assuming 'questions' is a list of dictionaries with 'name', 'options', and 'question' keys
    widgets_list = []
    responses = {}
    for el in questions:
        # Create an HTML widget for the question text
        question_text = el['question'].replace("\n", "<br>")
        question_widget = HTML(value=f"<b>{question_text}</b>")
        
        # Create the ToggleButtons for the options
        toggle_buttons = ToggleButtons(
            options=el['options'],
            disabled=False,
            value=None,
            button_style=''
        )
        
        # Combine the question and the buttons into a single VBox
        combined_widget = VBox([question_widget, toggle_buttons])
        
        # Add the combined widget to the list
        widgets_list.append(deepcopy(combined_widget))

        responses[el['name']] = deepcopy(combined_widget)

    # Display all the widgets
    display(VBox(widgets_list))
    """

    new_line = "\n"
    responses = {
        el['name']: VBox([HTML(value=f"<b>{el['question'].replace(new_line, '<br>')}</b>"), 
            ToggleButtons(
                options=el['options'],
                disabled=False,
                value=None,
                button_style=''
            )]) for el in questions
    }        

    
    for set, buttons in responses.items():
        display(buttons)
    
    btn = Button(description='submit', button_style='info', icon='check')
    def on_click(btn):
        nonlocal current_index
        labels_on = {}
        for cat, buttons in responses.items():
            labels_on[cat] = buttons.value
            buttons.value = None
        add_annotation(labels_on)
        render(current_index)
        
    btn.on_click(on_click)
    display(btn)
    render(current_index)
    return
    
