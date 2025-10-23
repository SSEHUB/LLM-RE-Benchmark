

# Standard Prompt
def build_prompt(text: str):

    prompt = f"""
    
    Context: The following text excerpt is taken from a requirements document and may contain information relevant to functional requirements.
    In the context of requirements engineering, functional requirements can be defined as follows: "{funcReq}"

    Process the following text excerpt according to the instruction.
    Text excerpt: "{text}"

    Instruction: 
    Identify all sentences that contain functional requirement information contained in the text excerpt.
    Output the identified sentences verbatim. This means that the exact wording should be retained.
    Sentence parts with information that does not belong to the actual functional requirement information can be removed from the sentence and should not be output.
    Output all identified sentences chronologically in a numbered list. One sentence should correspond to one entry in the list. The list should follow the following output structure. 
    If the text excerpt does not contain any functional requirement information, the following sentence is output: This text does not contain any functional requirement information.

    Output structure (example for a text containing two identified scentences): 
    1. <Placeholders for sentence one>
    2. <Placeholders for sentence two>

    Note: Return only sentences that contain functional requirement information according to the given structure and nothing else. Additional explanations are not desired. 
    Furthermore, no introductory or descriptive sentence should be added that is placed before or after the list of requirements. 

    """

    return prompt


funcReq = """Functional requirements define a function to be provided by the system or a system component. They describe actions to be performed by a system independently as well as interactions of the system with human users or systems and requirements for general, functional agreements."""

#funcReq = """Functional requirements are statements of services the system should provide, how the system should react to particular inputs and how the system should behave in particular situations. 
#They specify the functionality of the system to be developed or a service to be provided."""
