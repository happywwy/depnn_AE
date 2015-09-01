# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:28:35 2015

@author: wangwenya
"""

from dtree_util import *
import gen_util as gen
import sys, cPickle, random, os
from numpy import *

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None
    
wordnet_lemmatizer = WordNetLemmatizer()


f_test = open('data_semEval/raw_parses_restest0', 'r')
sentence_file = open('data_semEval/parsedSentence_restest0.txt', 'r')
data_test = f_test.readlines()
plist = []
tree_dict_test = []
rel_list = []

train_input = cPickle.load(open('data_semEval/final_input_res', 'rb'))
vocab = train_input[0]

label_file = open('data_semEval/aspectTerm_restest0', 'r')

for line in data_test:
    if line.strip():
        rel_split = line.split('(')
        rel = rel_split[0]
        deps = rel_split[1][:-1]
        deps = deps.replace(')','')
        if len(rel_split) != 2:
            print 'error ', rel_split
            sys.exit(0)

        else:
            dep_split = deps.split(',')
            
        if len(dep_split) > 2:
            fixed = []
            half = ''
            for piece in dep_split:
                piece = piece.strip()
                if '-' not in piece:
                    half += piece

                else:
                    fixed.append(half + piece)
                    half = ''

                    #print 'fixed: ', fixed
            dep_split = fixed

        final_deps = []
        for dep in dep_split:
            words = dep.split('-')
            word = words[0]
            ind = int(words[len(words) - 1])

            if len(words) > 2:
                word = '-'.join([w for w in words[:-1]])

            final_deps.append( (ind, word.strip()) )
            
        plist.append((rel,final_deps))

    else:
        max_ind = -1
        for rel, deps in plist:
            for ind, word in deps:
                if ind > max_ind:
                    max_ind = ind

        # load words into nodes, then make a dependency tree
        nodes = [None for i in range(0, max_ind + 1)]
        for rel, deps in plist:
            for ind, word in deps:
                nodes[ind] = word

        tree = dtree(nodes)
        
        sentence = sentence_file.readline().strip()
        terms = word_tokenize(sentence)
        pos_review = nltk.pos_tag(terms)
        
        word_pos = {}
        for item in pos_review:
            word_pos[item[0]] = penn_to_wn(item[1])
        
        aspect_term = label_file.readline().strip()
        
        if aspect_term != 'NIL':
            #aspect_term += " "
            line = aspect_term.split(' ')
            
            for term in nodes:
                if term in word_pos.keys():
                    ind = nodes.index(term)
                    tree.get(ind).word = term.lower()
                    
                    if term in line:
                        tree.get(ind).trueLabel = 1
                        
                    if word_pos[term] != None:
                        tree.get(ind).word = wordnet_lemmatizer.lemmatize(term.lower(), word_pos[term])
       
        """
        if aspect_term != '':
            aspect_term += ";"
            line = aspect_term.split(';')
            
            for element in line:
                if element != "":
                    element.split(',')
                    polarity = element[1]
                    aspect = element[0]
                    aspect += ' '
            
                    for term in nodes:
                        if term in aspect.split(' '):
                            ind = nodes.index(term)
                            tree.get(ind).trueLabel = 1        
        """
        # add dependency edges between nodes
        for rel, deps in plist:
            par_ind, par_word = deps[0]
            kid_ind, kid_word = deps[1]
            tree.add_edge(par_ind, kid_ind, rel)

        tree_dict_test.append(tree)  
        
        for node in tree.get_nodes():
            if node.word.lower() in vocab:
                
                node.ind = vocab.index(node.word.lower())
            
            for ind, rel in node.kids:
                if rel not in rel_list:
                    rel_list.append(rel)

        plist = []


cPickle.dump((rel_list, tree_dict_test), open("data_semEval/final_input_restest0", "wb"))



