

def consumeFile(nn,path):
            
    with open(path,"r") as text_file:
        lines=text_file.readlines
        consumeNeuralNet(nn, lines)
        
def consumeNeuralNet(nn, lines):
    y=0
    for l in range(0,len(nn.config.layer_sizes)-1):
        print("readling layer " + str(l) + " @" + str(y))
        y = consume2d(lines, y, nn.ws[l])
        y = consume1d(lines, y, nn.bs[l])
            
        

def consumeComments(lines,  y):
    while lines[y].startswith("#") or lines[y].isspace():
        print("skip comment @" + str(y) + ":" + lines[y])
        y+=1
    return y


def consume1d(lines,  y,  ds) : 
    y = consumeComments(lines, y)
    print("readling 1d @" + str(y)+":"+ lines[y])
    tokens = lines[y].split(",")
    y+=1
    for i in range(0,len(ds)):
        ds[i] = float(tokens[i])

    return y

        

def consume2d(content, y,  ds) : 
    print("readling 2d @" + str(y))
    for i in range(0,len(ds)):
        y = consume1d(content, y, ds[i])
    return y
        

def dump1d(vs,info=""):
    s=info+("\n" if info!="" else "")
    for i,v in enumerate(vs):    
        if i>0:
            s=s+', ' 
        s=s+str(v)  
    return s      

def dump2d(vss,info=""):
    s=info+("\n" if info!="" else "")
    for i,vs in enumerate(vss):    
        if i>0:
            s=s+'\n'
        s=s+dump1d(vs,"" if info=="" else (info+"["+str(i)+"]"))  
    return s      


def dump3d(vsss,info=""):
    s=info+("\n" if info!="" else "")
    for i,vss in enumerate(vsss):    
        if i>0:
            s=s+'\n\n'
        s=s+dump2d(vss,"" if info=="" else (info+"["+str(i)+"]"))  
    return s     

def dumpNeuralNet(nn):
    csv=""
    for l in range (0,len(nn.config.layer_sizes)-1):
        csv=csv+"\n# layer "+str(l)
        csv=csv+"\n"+dump2d(nn.ws[l],"#ws["+str(l)+"]")
        csv=csv+"\n"+dump1d(nn.bs[l],"#bs["+str(l)+"]")
    return csv
    