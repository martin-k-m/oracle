# Minimal GPT-2 byte-level BPE encode/decode, for validating the Twill runtime.
import json, os, sys, regex as re
D = sys.argv[1] if len(sys.argv) > 1 else "models/gpt2"
def bytes_to_unicode():
    bs=list(range(ord("!"),ord("~")+1))+list(range(ord("\xa1"),ord("\xac")+1))+list(range(ord("\xae"),ord("\xff")+1))
    cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return {b:chr(c) for b,c in zip(bs,cs)}
B2U=bytes_to_unicode(); U2B={v:k for k,v in B2U.items()}
vocab=json.load(open(os.path.join(D,"vocab.json")))
dec={v:k for k,v in vocab.items()}
merges=[tuple(l.split()) for l in open(os.path.join(D,"merges.txt")).read().split("\n")[1:] if l and len(l.split())==2]
rank={m:i for i,m in enumerate(merges)}
PAT=re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
def bpe(tok):
    word=list(tok)
    while len(word)>1:
        pairs={(word[i],word[i+1]) for i in range(len(word)-1)}
        best=min(pairs,key=lambda p:rank.get(p,1e18))
        if best not in rank: break
        a,b=best; nw=[]; i=0
        while i<len(word):
            if i<len(word)-1 and word[i]==a and word[i+1]==b: nw.append(a+b); i+=2
            else: nw.append(word[i]); i+=1
        word=nw
    return word
def encode(text):
    ids=[]
    for tok in PAT.findall(text):
        s="".join(B2U[b] for b in tok.encode("utf-8"))
        ids+=[vocab[p] for p in bpe(s)]
    return ids
def decode(ids):
    s="".join(dec[i] for i in ids)
    return bytearray(U2B[c] for c in s).decode("utf-8", errors="replace")
if __name__=="__main__":
    cmd=sys.argv[2]
    if cmd=="enc": print(" ".join(map(str,encode(sys.argv[3]))))
    else: print(decode([int(x) for x in sys.argv[3].split()]))
