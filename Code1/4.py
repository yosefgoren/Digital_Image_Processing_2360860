#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
from skimage.color import rgb2gray

#QUESTION 4 - COMPRESSION

#section a:
heisenberg= plt.imread("heisenberg.jpg")
h_entropy=shannon_entropy(heisenberg,base=2)

print(f"The entropy of Heisenberg is: {h_entropy}")

#section b:

# Below is a github implementing of a huffman encoder for text, with some changes so that it will work for a grayscale image (though we still encode using text).
# source: https://github.com/ybruce61414/Huffman-Code/blob/master/HuffmanCode.ipynb

class Huffman_node():
    def __init__(self,cha,freq):
        self.cha = cha
        self.freq = freq
        self.Lchild = None
        self.Rchild = None
   
    def __repr__(self):
            return '(node object %s:%d)' % (self.cha,self.freq)
        
class HuffmanCoding():
    def __init__(self,text):
        self.root = None
        self.text = text
        self.nodedic = {}
        self.huffcodes = {}
        self.encodes = []
        self.decodes = []
                
    #------ generating huffman tree -------   
    def generate_tree(self):
        self.generate_node() 
        while len(self.nodedic) != 1:
            min_node1 = self.find_minNode()
            min_node2 = self.find_minNode()
            self.root = self.merge_nodes(min_node1,min_node2)
        return self.root              
        
    #---- function set for generating huffman tree -----
    def character_freq(self):
        #generate dic-{cha:freq}
        count = {}
        for cha in self.text:
            count.setdefault(cha,0)
            count[cha] += 1
        return count     

    def generate_node(self):
        #generate dic-{freq:node}
        c = self.character_freq()
        #storing each cha & freq into huffmanNode
        for k,v in c.items():
            newnode = Huffman_node(k,v)
            #multiple value for the same key
            #dic-{key:[ob1,ob2..]}
            self.nodedic.setdefault(v,[]).append(newnode)
        return self.nodedic
    
    def find_minNode(self):
        keys = list(self.nodedic.keys())
        minkey, minlist = keys[0], self.nodedic[keys[0]]
        for k,v in self.nodedic.items():
            if minkey > k:
                minkey,minlist = k,v
        minvalue = minlist.pop(0)
        if not minlist:
            #empty list,delete the minNode from dic
            del self.nodedic[minkey]    
        #return minNode object
        return minvalue 
    
    def merge_nodes(self,min1,min2):
        newnode = Huffman_node(None,min1.freq + min2.freq)
        newnode.Lchild,newnode.Rchild = min1,min2
        #adding newnode into self.nodedic
        self.nodedic.setdefault(min1.freq + min2.freq,[]).append(newnode) 
        return newnode
    
    #----------generating huffman code-----------
    def generate_huffcode(self):
        code = ''
        if self.root != None:
            return self.rec_generate_huffcode(self.root,code)         
            
    def rec_generate_huffcode(self,cur_node,codestr):
        if not cur_node.Lchild and not cur_node.Rchild:
            self.huffcodes[cur_node.cha] = codestr  
        if cur_node.Lchild:
            self.rec_generate_huffcode(cur_node.Lchild,codestr + '0')
        if cur_node.Rchild:
            self.rec_generate_huffcode(cur_node.Rchild,codestr + '1')
         
    #----------------compression-------------------
    def encode(self):
        for cha in self.text:
            self.encodes.append(self.huffcodes[cha])
        #strings in list merge into one string    
        self.encodes = ''.join(self.encodes)
        #turn encodes into string
        return self.encodes     
        
    #----------------decompression------------------
    def decode(self):
        temp_str,temp_dic = '',{}
        #reverse huffcodes
        for k,v in self.huffcodes.items():
            temp_dic[v] = k
        
        for binary_code in self.encodes:
            temp_str += binary_code
            if temp_str in temp_dic.keys():
                self.decodes.append(temp_dic[temp_str])
                temp_str = ''
        self.decodes = ''.join(self.decodes)         
        return self.decodes 

def huffman_encoder(im):
    #the github huffman encoder uses text, so we need to convert the image to a utf format.
    flatten_im = im.flatten().astype(np.int16) #int16 explained below, int8 is insufficient for difference encoding
    input = flatten_im.tobytes().decode("utf-16-le") #decode image into text for huffman encoder, we use np.int16 and utf-16 so that it will handle values from (-255,255), in total 511 bits
    huffman = HuffmanCoding(input)
    huffman.generate_tree()
    huffman.generate_huffcode()
    code = huffman.encode()
    dict = huffman.huffcodes
    char_freq = huffman.character_freq()
    avglen = sum(len(dict[character])*freq for character,freq in char_freq.items())/len(input)
    return code,dict,avglen

# section c
heisenberg_code,heisenberg_dict,heisenberg_avglen = huffman_encoder(heisenberg)
# code length:
heis_codelength = len(heisenberg_code)
print(f"Heisenberg Huffman code length is {heis_codelength}")
# our original image is 8 bits, so compression ratio is 8/avglen
heis_compression_ratio = 8/heisenberg_avglen
print(f"Heisenberg Huffman compression ratio is {heis_compression_ratio}")

#section d

def huffman_decoder(code,dict,width,height):
    huffman = HuffmanCoding("")
    huffman.encodes=code
    huffman.huffcodes=dict
    output=huffman.decode()
    output_flat = output.encode("utf-16-le")
    output_numpy=np.frombuffer(output_flat,dtype=np.int16) # again, we decode using int16 since we encoded with it
    return output_numpy.reshape((width,height))

decoded_heisenberg = huffman_decoder(heisenberg_code,heisenberg_dict,heisenberg.shape[0],heisenberg.shape[1])
heisenberg_mse = np.mean((decoded_heisenberg-heisenberg)**2)
print(f"MSE between decoded image and original image is: {heisenberg_mse}")

# section e

mauritius_rgb= plt.imread("mauritius.jpg")
mauritius =(rgb2gray(mauritius_rgb)*255).astype(np.uint8) #because rgb2gray produces floats between 0 and 1, we want greyscale between 0 and 255

mauritius_entropy=shannon_entropy(mauritius,base=2)
print(f"The entropy of Mauritius is: {mauritius_entropy}")
mauritius_code,mauritius_dict,mauritius_avglen = huffman_encoder(mauritius)
print(f"The information rate of Mauritius is: {mauritius_avglen}")

# section f
scotland_rgb = plt.imread("scotland.jpg")
scotland = (rgb2gray(scotland_rgb)*255).astype(np.int16) # because rgb2gray produces floats between 0 and 1, we want greyscale between 0 and 255
scot_column_stack = scotland.flatten(order='F')
def zigzag_ver2(img):
  zigzaged = np.concatenate([np.diagonal(img[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-img.shape[0], img.shape[0]+abs(img.shape[0]-img.shape[1])+1)])
  return zigzaged

def un_zigzag(vec_img, M, N):
  output = np.zeros((M,N))
  assert M*N == len(vec_img), f"A vector of size {len(vec_img)} cannot be rearranged into an ({M},{N}) matrix."
  # create indices vector
  indices = np.arange(len(vec_img))
  # rearrange the indices the same way as the original matrix
  indices = indices.reshape((M, N))
  indices = zigzag_ver2(indices)
  # for each element in the vector, replace in the original matrix index
  for (k,element) in zip(indices,vec_img):
    i = int(np.floor(k/N)) # row
    j = int(np.mod(k,N)) # col
    output[i,j]=element # placement
  return output

scot_zigzag = zigzag_ver2(scotland)
column_diff = np.zeros_like(scot_column_stack,dtype=np.int16)
column_diff[0] = scot_column_stack[0]
zigzag_diff = np.zeros_like(scot_zigzag,dtype=np.int16)
zigzag_diff[0] = scot_zigzag[0]
column_diff[1:]=[scot_column_stack[i]-scot_column_stack[i-1] for i in range(1,len(scot_column_stack))]
zigzag_diff[1:]=[scot_zigzag[i]-scot_zigzag[i-1] for i in range(1,len(scot_zigzag))]
column_image = column_diff.reshape(scotland.shape,order='F')
zigzag_image = un_zigzag(zigzag_diff,scotland.shape[0],scotland.shape[1])
column_hist = np.histogram(column_image, bins=511,range=(-255,256))[0]
zigzag_hist = np.histogram(zigzag_image, bins=511,range=(-255,256))[0]

#plotting the histograms
plt.figure()
plt.bar(np.arange(-255, 256),column_hist,width=1)
plt.title("Column Difference")
plt.xlabel("Difference")
plt.ylabel("Amount")
plt.show()

plt.figure()
plt.bar(np.arange(-255,256),zigzag_hist,width=1)
plt.title("Zigzag Difference")
plt.xlabel("Difference")
plt.ylabel("Amount")
plt.show()

column_code, column_dict, column_avglen = huffman_encoder(column_image)
zigzag_code, zigzag_dict, zigzag_avglen = huffman_encoder(zigzag_image)
print(f"avglen of column stacked image is {column_avglen}")
print(f"avglen of zigzag image is {zigzag_avglen}")