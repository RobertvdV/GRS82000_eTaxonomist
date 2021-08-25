def random_text_splitter_old(text):
    
    import random
    
    """
    Random breaks up a text into an X amount of 
    sentences. The output sentences consist of 
    a minimum of 10 sentences.
    """

    # Dont split short sentences
    if len(text.split()) <= 10:
        return [text]

    # Split text
    words = text.split()
    # Get length
    sentlength = len(words)
    # Random int
    randomint = random.randint(10, sentlength)
    # Check sentences from text
    parts = sentlength // randomint

    # Create sentences
    sentences = [' '.join(words[randomint*i:randomint*(i+1)]) for i in range(0, parts)]
    last_part = ' '.join(words[randomint*parts:])
    if len(last_part.split()) <= 10:
        sentences[-1] = sentences[-1] + last_part
    else:
        sentences += [last_part]

    return sentences

def random_text_splitter(text):
    
    import random
    
    """
    Random breaks up a text into an X amount of sentences. 
    The output sentences consist of a minimum of 10 sentences.
    """

    # Split text
    words = text.split()
    # Get the amount of words
    word_amount = len(words)
    # Create counter
    remaining_word_amount = word_amount
    # Init list
    parts = []
    # While words remaining
    while remaining_word_amount > 0:
        if len(words) < 10:
            # Add last part if less then 10
            parts[-1] = parts[-1] + ' '.join(words)
            # exit
            remaining_word_amount = 0
        # Generate random int
        randint = random.randint(10, 50)
        # Append to list 
        parts.append(' '.join(words[:randint]))
        # Delete previous selection
        words = words[randint:]
        # Update counter
        remaining_word_amount -= randint
        
    return parts