# Step 1: Importing the PyPDF2 library
from PyPDF2 import PdfMerger

# Step 2: Creating an instance of PdfMerger
merger = PdfMerger()

# Step 3: List of PDF files to be merged
pdf_files = ['basic-text.pdf', 'fillable-form.pdf']  # Add your PDF file names here

# Step 4: Iterating through the list of PDF files and appending them to the merger
for pdf in pdf_files:
    merger.append(pdf)

# Step 5: Writing the merged PDF to a new file
output_pdf = 'merged_output.pdf'  # Specify the name of the output PDF file
merger.write(output_pdf)

# Step 6: Closing the PdfMerger object to free resources
merger.close()

# Step 7: Displaying a message to indicate success
print(f"The PDFs have been successfully merged into '{output_pdf}'")
