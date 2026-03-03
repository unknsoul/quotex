import docx2txt

if __name__ == '__main__':
    text = docx2txt.process('c:\\Users\\hardi\\Downloads\\quotex\\QUOTEX_LORD_v10_FinalPlan.docx')
    with open('QUOTEX_LORD_v10_FinalPlan.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Done")
