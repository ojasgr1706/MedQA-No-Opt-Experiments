import argparse
import fitz
from operator import itemgetter
import re

def none_or_str(val):
    if val == "None":
        return None
    return int(val)

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", help="File name to extract", required=True, type=str)
parser.add_argument("--output_file", help="Output file name", required=True, type=str)
parser.add_argument("--top_k", help="If you want to extract text which is in the top_k most occuring font sizes", required=False, type=none_or_str, default=None)
parser.add_argument("--start_pages", help="start page numbers for extraction (Single number or list of all start pages in format : [a,b,c,..])", required=True)
parser.add_argument("--end_pages", help="end page numbers for extraction (Single number or list of all end pages in format : [a,b,c,..])", required=True)
parser.add_argument("--annotations", help="Output file with annotations (font sizes) or not", required=False, type=bool, default=False)
args = parser.parse_args()

# print(args)

def fonts(doc, granularity=False, top_k = None):
    """Extracts fonts and their usage in PDF documents.
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param granularity: also use 'font', 'flags' and 'color' to discriminate text
    :type granularity: bool
    :rtype: [(font_size, count), (font_size, count}], dict
    :return: most used fonts sorted by count, font style information
    """
    styles = {}
    font_counts = {}

    for i in range(len(start)):
        for page in doc.pages(start[i],end[i]):
            blocks = page.get_text("dict", sort = True)["blocks"]
            for b in blocks:  # iterate through the text blocks
                if b['type'] == 0:  # block contains text
                    for l in b["lines"]:  # iterate through the text lines
                        for s in l["spans"]:  # iterate through the text spans
                            if granularity:
                                identifier = "{0}_{1}_{2}_{3}".format(s['size'], s['flags'], s['font'], s['color'])
                                styles[identifier] = {'size': s['size'], 'flags': s['flags'], 'font': s['font'],
                                                    'color': s['color']}
                            else:
                                identifier = "{0}".format(s['size'])
                                styles[identifier] = {'size': s['size'], 'font': s['font']}

                            font_counts[identifier] = font_counts.get(identifier, 0) + len(s['text'].split())  # count the fonts usage in words

    font_counts = sorted(font_counts.items(), key=itemgetter(1), reverse=True)

    if len(font_counts) < 1:
        raise ValueError("Zero discriminating fonts found!")

    if top_k:
        font_counts = font_counts[:top_k]

    return font_counts, styles

def font_tags(font_counts, styles):
    """Returns dictionary with font sizes as keys and tags as value.
    :param font_counts: (font_size, count) for all fonts occuring in document
    :type font_counts: list
    :param styles: all styles found in the document
    :type styles: dict
    :rtype: dict
    :return: all element tags based on font-sizes
    """
    p_style = styles[font_counts[0][0]]  # get style for most used font by count (paragraph)
    p_size = p_style['size']  # get the paragraph's size

    # sorting the font sizes high to low, so that we can append the right integer to each tag 
    font_sizes = []
    for (font_size, count) in font_counts:
        font_sizes.append(float(font_size))
    font_sizes.sort(reverse=True)

    # aggregating the tags for each font size
    idx = 0
    size_tag = {}
    for size in font_sizes:
        idx += 1
        if size == p_size:
            idx = 0
            size_tag[size] = '<p>'
        if size > p_size:
            size_tag[size] = '<h{0}>'.format(idx)
        elif size < p_size:
            size_tag[size] = '<s{0}>'.format(idx)

    return size_tag

# final line-separated text with size tags

def headers_para_tag(doc, size_tag):
    """Scrapes headers & paragraphs from PDF and return texts with element tags.
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param size_tag: textual element tags for each size
    :type size_tag: dict
    :rtype: list
    :return: texts with pre-prended element tags
    """
    header_para = []  # list with headers and paragraphs
    first = True  # boolean operator for first header
    previous_s = {}  # previous span

    for i in range(len(start)):
        for page in doc.pages(start[i],end[i]):
            blocks = page.get_text("dict", sort = True)["blocks"]
            for b in blocks:  # iterate through the text blocks
                if b['type'] == 0:  # this block contains text

                    # REMEMBER: multiple fonts and sizes are possible IN one block

                    block_string = ""  # text found in block
                    for l in b["lines"]:  # iterate through the text lines
                        for s in l["spans"]:  # iterate through the text spans
                            try:
                                if s['size'] in size_tag:
                                    size = size_tag[s['size']]
                                    if s['text'].strip():  # removing whitespaces:
                                        if first:
                                            # print("first")
                                            previous_s = s
                                            first = False
                                            block_string = {size : s['text']}
                                        else:
                                            if s['size'] == previous_s['size']:
                                                # print("a")

                                                if block_string and all((c == "|") for c in block_string):
                                                    # print("b")
                                                    # block_string only contains pipes
                                                    block_string = {size : s['text']}
                                                if block_string == "":
                                                    # print('c')
                                                    # new block has started, so append size tag
                                                    block_string = {size : s['text']}
                                                else:  # in the same block, so concatenate strings
                                                    # print('d')
                                                    block_string[size] += " " + s['text']

                                            elif len(block_string) > 0:
                                                # print('e', block_string)

                                                header_para.append(block_string)
                                                # print("s['size'] = ",s['size'])
                                                block_string = {size : s['text']}
                                                # print(block_string)

                                            previous_s = s
                            except:
                                continue

                    if type(block_string) == dict and len(block_string) > 0:
                        header_para.append(block_string)

    return header_para
    # return 0

# final concatenated text without size tags

def headers_para(doc, size_tag):
    """Scrapes headers & paragraphs from PDF and return texts with element tags.
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param size_tag: textual element tags for each size
    :type size_tag: dict
    :rtype: list
    :return: texts with pre-prended element tags
    """
    header_para = ""  # string with paragraphs
    first = True  # boolean operator for first header
    previous_s = {}  # previous span

    for i in range(len(start)):
        for page in doc.pages(start[i],end[i]):
            blocks = page.get_text("dict", sort = True)["blocks"]
            for b in blocks:  # iterate through the text blocks
                if b['type'] == 0:  # this block contains text

                    # REMEMBER: multiple fonts and sizes are possible IN one block

                    block_string = ""  # text found in block
                    for l in b["lines"]:  # iterate through the text lines
                        for s in l["spans"]:  # iterate through the text spans
                            try:
                                # modified the code to include chemical formula numbers
                                # chem formulas have numbers below a certain font size, using '6' for now
                                if s['size'] in size_tag or s['size'] <= 6:
                                    if s['text'].strip():  # removing whitespaces:
                                        if first:
                                            # print("first")
                                            previous_s = s
                                            first = False
                                            block_string = s['text']
                                        else:
                                            if s['size'] == previous_s['size']:
                                                # print("a")

                                                if block_string and all((c == "|") for c in block_string):
                                                    # print("b")
                                                    # block_string only contains pipes
                                                    block_string = s['text']
                                                if block_string == "":
                                                    # print('c')
                                                    # new block has started, so append size tag
                                                    block_string = s['text']
                                                else:  # in the same block, so concatenate strings
                                                    # print('d')
                                                    block_string += " " + s['text']

                                            elif len(block_string) > 0:
                                                # print('e', block_string)

                                                header_para += block_string
                                                # print("s['size'] = ",s['size'])
                                                block_string = s['text']
                                                # print(block_string)

                                            previous_s = s
                            except:
                                continue

                    if len(block_string) > 0:
                        header_para += block_string + " \n"

    return header_para
    # return 0

def main():

    start_pages = args.start_pages
    end_pages = args.end_pages

    global start
    global end
    start = []
    end = []

    if re.search(r"^\[+",start_pages):
        start = list(map(int,start_pages.strip("[]").split(',')))
        end = list(map(int,end_pages.strip("[]").split(',')))
    else:
        start = [int(start_pages)]
        end = [int(end_pages)]

    doc = fitz.open(args.input_file)
    font_counts, styles = fonts(doc, granularity=False, top_k=args.top_k)
    print("font_counts : ",font_counts, "\n")
    # print("styles : ",styles, "\n")
    size_tag = font_tags(font_counts,styles)
    # print("size_tag : ",size_tag, "\n")
    final_text = headers_para(doc,size_tag)
    print("FINAL TEXT (first few lines) ---------------------------- \n")
    print(final_text[:500])
    
    # This line will remove all newline characters, this will allow the training script to take the whole document as part of the same example, and feed into the model in a sliding window fashion.
    continuous_text = ""
    for text in final_text:
        continuous_text += re.sub(r"\n+", " ", text)

    if(input("\nFULL: ") == "y"):
        print("FINAL TEXT (full) ---------------------------- \n")
        print(continuous_text)

    if(input("\nSAVE: ") == "y"):
        with open(args.output_file,"w+") as f:
            f.write(continuous_text)

if __name__ == "__main__":
    main()