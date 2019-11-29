import spacy

nlp = spacy.load('en_core_web_sm')

doc = nlp('srm goes to market. he eats apple')

for t in doc:
    print(t.text, t.pos_, t.dep_, t.shape_, spacy.explain(t.dep_))
    
for s in doc.sents:
    print(s)
    
doc[5].is_sent    


doc = nlp('Apple and oranges')    

for e in doc.ents:
    print(e)
    print(spacy.explain(e.label_))
    

doc = nlp('Apple is wing 6c')        

for t in doc:
    print(t, t.lemma_, t.shape_)
    


len(nlp.Defaults.stop_words)

nlp.Defaults.stop_words.remove('yet')
nlp.vocab['yet'].is_stop=False

doc = nlp(u'john is looking at them.')
for i in doc.sents:
    print(i)

print(doc[2].is_sent_start)

for i in doc:
    print(i, i.pos_, i.lemma_)


doc4 = nlp(u"Let's visit St. Louis in the U.S. next year.")

for t in doc4:
    print(t)        

    
    
doc5 = nlp(u'It is better to give than to receive.')

print(doc5[-4:])

from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)

p1 = [{'LOWER':'solarpower'}]
p2 = [{'LOWER':'solar'}, {'LOWER':'power'}]
p3 = [{'LOWER':'solar'},{'IS_PUNCT':True, 'OP':'*'},{'LOWER':'power'}]

matcher.add('sp',None, p1,p3)

doc = nlp(u'The Solar Power industry continues to grow as demand \
for solarpower increases. Solar-power cars are gaining popularity.')    

found_matches = matcher(doc)
print(found_matches)    

for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]  # get string representation
    span = doc[start:end]                    # get the matched span
    print(match_id, string_id, start, end, doc[start:end])
    
    
from spacy.matcher import PhraseMatcher

p_matcher = PhraseMatcher(nlp.vocab)    

p_list=['solarpower']
p_pattern = [nlp(p_list[0])]
p_matcher.add('psp',None,*p_pattern)
found_matches = p_matcher(doc)
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]  # get string representation
    span = doc[start:end]                    # get the matched span
    print(match_id, string_id, start, end, doc[start-5:end+5]) 
    
    
    
    
    
    
#########################################################################
##########          Assessment     ##################    

import spacy
nlp = spacy.load('en_core_web_sm')    

with open('D:\ML\ML_Rev\Datasets\owlcreek.txt') as f:
    doc = nlp(f.read())

doc[:36]    

len(doc)

sents = [s for s in doc.sents]
len(list(sents))

print(sents[2])

for token in sents[2]:
    print(f'{token.text:{15}} {token.pos_:{5}} {token.dep_:{10}} {token.lemma_:{15}}')
    
    
from spacy.matcher import Matcher
mat = Matcher(nlp.vocab)

p1 = [{'LOWER':'swimming'},{'IS_SPACE': True, 'OP':'*'}, {'LOWER':'vigorously'}]

mat.add('sm',None, p1)  

found_matches = mat(doc)
print(found_matches)   

for match_id, start, end in found_matches:
    print(doc[start-5:end+5])

for fm in found_matches:    
    for s in sents:
        if fm[1] < s.end:
            print(s, end='\n\n')            
            break

