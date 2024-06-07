import json

def datalist2scp(datalist, scp):
    with open(datalist, 'r') as rf, open(scp, 'w') as wf:
        for line in rf:
            line = line.strip()
            data = json.loads(line)
            path = data['wav']
            id = path.split('/')[-1].split('.')[0]
            wf.write(id + ' ' + path + '\n')

if __name__ == '__main__':
    datalist = 'data/train/data.list'
    scp = 'data/train/wav.scp'
    datalist2scp(datalist, scp)