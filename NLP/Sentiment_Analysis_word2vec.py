import spacy

nlp = spacy.load('en_core_web_md')

doc = nlp('lion')
doc.vector

doc = nlp('Lion cat Tiger dog mouse')

for tok1 in doc:
    for tok2 in doc:
        print(tok1.text + '---' +tok2.text+ '---'+str(tok1.similarity(tok2)))


from scipy import spatial    

cos_similarity = lambda x,y:1-spatial.distance.cosine(x, y)

king = nlp('king').vector
man = nlp('man').vector
woman = nlp('woman').vector

#n_vec = king-man+woman
n_vec = king-man+woman

sim_list = []

for w in nlp.vocab:
    if w.has_vector and w.is_lower and w.is_alpha:
        sim = cos_similarity(w.vector, n_vec)
        sim_list.append((w, sim))


sim_list = sorted(sim_list, key= lambda item: -item[1])        
print([w[0].text for w in sim_list[:10]])