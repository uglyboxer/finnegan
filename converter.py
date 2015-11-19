import csv

def file_make(idx):
    lst = []
    file_name = 'digits_' + str(idx) + '.txt'
    with open(file_name, 'r') as f:
        for line in f:
            lst.append(line.strip())


    write_name = 'digits_submission_' + str(idx) + '.csv'
    with open(write_name, 'w', newline='\n') as w:
        
        w.write('"ImageId","Label"')
        w.write('\n')
        writer = csv.writer(w)

        for idx, elem in enumerate(lst):
            row = idx+1, str(elem)
            writer.writerow(row)

for idx in range(5, 12):
    file_make(idx)
