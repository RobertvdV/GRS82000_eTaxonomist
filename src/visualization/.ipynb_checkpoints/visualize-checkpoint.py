def VisualizeDoc(sentences, per_sentence=True, save=False):
    
    from matplotlib import cm
    import spacy
    from spacy import displacy
    from spacy.matcher import PhraseMatcher
    from spacy.tokens import Span
    from spacy.util import filter_spans
    
    """
    Creates and HTML file (can be rendered in a notebook) by using the SpaCy 
    Displacy.
    
    per_sentence: Returns the visualization per sentence instead of a whole doc.
    save:         If True returns the html string to save.
    """
    
    # Init color map
    cmap = cm.get_cmap('Spectral')
    # Init color dict
    colors = {}
    # Init option dict
    options = {"ents": [],
               "colors": colors}
    # Init matcher
    matcher = PhraseMatcher(nlp.vocab)
    # Loop over the sentences
    for idx, sentence in enumerate(sentences):
        
        # Get the prediction values    
        prediction = SpanPredictor(str(sentence), pred_values=True)[1][1].numpy().item()
        
        # String ID            
        #text = '#{0} - {1:.2f}'.format(idx, prediction)
        text = f'{prediction:.3f}'
        # Add the patterns        
        pattern = nlp(str(sentence))
        matcher.add(text, None, pattern)

        # Colorize the strings
        if prediction > .5:
            colors[text] = matplotlib.colors.rgb2hex(cmap(prediction))
        else:
            colors[text] = matplotlib.colors.rgb2hex(cmap(prediction)) + '60'
        # Add the new ENTS to the doc
        options["ents"].append(text)

    # Match the enitities in the doc
    matches = matcher(doc)
    # Reset the current ENTS
    doc.ents = ()
    # Loop over the matches
    for match_id, start, end in matches:
        # Add the sentencen as a ENT
        span = Span(doc, start, end, label=match_id)
        #doc.ents = filter_spans(doc.ents)
        try:
            doc.ents = list(doc.ents) + [span]
        except:
            continue
            
    # Set title
    #doc.user_data["title"] = "Description Predictor"
    sentence_spans = list(doc.sents)
    
    if save and per_sentence:
        return displacy.render(sentence_spans, style='ent', options=options)
    elif save and not per_sentence:
        return displacy.render(doc, style='ent', options=options)
    elif not save and per_sentence:
        displacy.render(sentence_spans, style='ent', options=options)
    elif not save and not per_sentence:
        displacy.render(doc, style='ent', options=options)