#!/usr/bin/env python3
"""Build editable Word manuscripts. Requires python-docx, latex2mathml, mathml2omml.

Run from any directory. Render the outputs and inspect every page before release.
Performance figures are intentionally not fabricated by this builder.
"""
import json
from pathlib import Path
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
import latex2mathml.converter
import mathml2omml

ROOT = Path(__file__).resolve().parent


def create(title):
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.top_margin = section.bottom_margin = Inches(.75)
    section.left_margin = section.right_margin = Inches(.8)
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(11)
    normal.paragraph_format.line_spacing = 1.1
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.widow_control = True
    for name, size in [('Title', 18), ('Heading 1', 14), ('Heading 2', 12)]:
        style = doc.styles[name]
        style.font.name = 'Times New Roman'
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(11)
        style.paragraph_format.space_after = Pt(6)
    for style in doc.styles:
        for border in style.element.xpath('.//w:pBdr'):
            border.getparent().remove(border)
        for fonts in style.element.xpath('.//w:rFonts'):
            for key in ('asciiTheme', 'hAnsiTheme', 'eastAsiaTheme', 'cstheme'):
                fonts.attrib.pop(qn('w:' + key), None)
            fonts.set(qn('w:ascii'), 'Times New Roman')
            fonts.set(qn('w:hAnsi'), 'Times New Roman')
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    doc.core_properties.title = title
    doc.core_properties.author = 'Bofeng Duan et al.'
    return doc


def paragraph(doc, text, style=None):
    p = doc.add_paragraph(style=style)
    for chunk in re.split(r'(\[PENDING[^\]]*\])', text):
        run = p.add_run(chunk)
        if chunk.startswith('[PENDING'):
            run.font.color.rgb = RGBColor.from_string('9C3C17')
    return p


def equation(doc, latex):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(8)
    # mathml2omml 0.0.2 has malformed groupChr output for stretchy widetilde.
    # The ordinary tilde accent conveys the same notation and exports correctly.
    omml = mathml2omml.convert(latex2mathml.converter.convert(latex.replace('\\widetilde', '\\tilde')))
    element = parse_xml('<root xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">' + omml + '</root>')
    for operator in element.findall('.//' + qn('m:nary')):
        props = operator.find(qn('m:naryPr'))
        for name in ('sup', 'sub'):
            value = operator.find(qn('m:' + name))
            if value is not None and not ''.join(value.itertext()).strip():
                hidden = OxmlElement('m:' + name + 'Hide')
                hidden.set(qn('m:val'), '1')
                props.append(hidden)
    for child in list(element):
        p._p.append(child)


def table(doc, lines):
    rows = [[cell.strip() for cell in line.strip('|').split('|')] for line in lines if not re.match(r'^\|[ :|\-]+\|$', line)]
    t = doc.add_table(rows=1, cols=len(rows[0]))
    t.style = 'Table Grid'
    t.autofit = False
    for index, values in enumerate(rows):
        cells = t.rows[0].cells if index == 0 else t.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = value
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(4)
                for run in p.runs:
                    run.font.size = Pt(10)
                    run.bold = index == 0
        props = t.rows[index]._tr.get_or_add_trPr()
        props.append(OxmlElement('w:cantSplit'))
        if index == 0:
            props.append(OxmlElement('w:tblHeader'))
    doc.add_paragraph()


def manuscript():
    doc = create('AMORE revised manuscript')
    lines = (ROOT / 'manuscript.md').read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].startswith('|'):
                block.append(lines[i]); i += 1
            table(doc, block)
            continue
        if line.startswith('# '):
            paragraph(doc, line[2:], 'Title')
        elif line.startswith('## '):
            paragraph(doc, line[3:], 'Heading 1')
        elif line.startswith('### '):
            paragraph(doc, line[4:], 'Heading 2')
        elif line.startswith('EQ: '):
            # Split the two longest compound displays into independently editable equations.
            content = line[4:]
            separators = r',\qquad ' if content.startswith('q_i=') else r',\quad ' if content.startswith('Z=') else None
            for part in content.split(separators) if separators else [content]:
                equation(doc, part.rstrip(','))
        elif line:
            paragraph(doc, line)
        i += 1
    out = ROOT / 'manuscript/AMORE_revised.docx'
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    return out


def response():
    doc = create('Response to reviewers: IPM-D-26-07304')
    paragraph(doc, 'Response to reviewers', 'Title')
    paragraph(doc, 'Manuscript IPM-D-26-07304 | AMORE', 'Subtitle')
    paragraph(doc, 'We thank the editor and reviewers for their detailed assessment. We agree that the submitted evidence does not establish consistent superiority or validate the shared-affinity mechanism. The revision corrects the implementation description, narrows unsupported claims, and defines controlled comparisons on the original AMORE backbone. Responses below distinguish completed specification/code changes from experimental evidence still required before submission.')
    paragraph(doc, '[PENDING E01–E08: replace every evidence slot with completed results and exact table/section references before submitting this response. Cross-domain evaluation is not added, and four requested related-method comparisons remain open.]')
    entries = json.loads((ROOT / 'reviewer_matrix.json').read_text())
    md = ['# Response to reviewers: IPM-D-26-07304', '']
    locations = {
        'R1.1': 'Sections 4.2 and 5.1', 'R1.2': 'Sections 3.2, 4.3 and 5.2', 'R1.3': 'Sections 2 and 4.2',
        'R1.4': 'Sections 4.1, 6 and 7', 'R1.5': 'Sections 3.6, 4.5 and 5.3',
        'R2.1': 'Sections 3.2–3.3 and 4.3', 'R2.2': 'Section 3 and Algorithm 1', 'R2.3': 'Section 4.2',
        'R2.4': 'Sections 4.4 and 5.3', 'R2.5': 'Sections 3–5', 'R3.1': 'Sections 1–2',
        'R3.2': 'Sections 3.4 and 4.3', 'R3.3': 'Sections 4.2 and 5.1', 'R3.4': 'Sections 4.3 and 5.2',
        'R3.5': 'Sections 4.3 and 5.2', 'R3.6': 'Sections 4.4 and 5.3', 'R3.7': 'Sections 4.5 and 5.3',
        'R3.8': 'Sections 1, 3 and 6–7'}
    previous = None
    for entry in entries:
        reviewer = entry['id'].split('.')[0]
        if previous != reviewer:
            if previous:
                doc.add_page_break()
            paragraph(doc, 'Reviewer ' + reviewer[1:], 'Heading 1')
            previous = reviewer
        paragraph(doc, entry['id'] + ' | ' + entry['concern'], 'Heading 2')
        paragraph(doc, 'Comment summary (paraphrased): ' + entry['concern'])
        paragraph(doc, entry['response'])
        paragraph(doc, 'Location: ' + locations[entry['id']] + '. Evidence/protocol: ' + entry['experiments'] + '.')
        md.extend(['## ' + entry['id'] + ': ' + entry['concern'], '', entry['response'], '',
                   '**Location:** ' + locations[entry['id']], '', '**Status:** ' + entry['status'], ''])
    out = ROOT / 'response/Response_to_reviewers.docx'
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    (ROOT / 'response.md').write_text('\n'.join(md))
    return out


if __name__ == '__main__':
    print(manuscript())
    print(response())
