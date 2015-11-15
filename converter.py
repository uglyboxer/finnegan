import csv

lst = []
with open('digits.txt', 'r') as f:
    for line in f:
        lst.append(line.strip())


with open('digits_submission.csv', 'w', newline='\n') as w:
    
    w.write('"ImageId","Label"')
    w.write('\n')
    writer = csv.writer(w)

    for idx, elem in enumerate(lst):
        row = idx+1, str(elem)
        writer.writerow(row)
