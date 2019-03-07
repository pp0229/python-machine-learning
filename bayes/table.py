import matplotlib.pyplot as plt

# The data for the table
rows = ('Size', 'Weight', 'Color')
columns = ('Apple', 'Blueberry', 'Coconut')
cells = [['Moderate', 'Small', 'Large'],
         ['Moderate', 'Light', 'Heavy'],
         ['Red', 'Blue', 'Brown']]

# Create table
table = plt.table(cellText=cells,
                  rowLabels=rows,
                  colLabels=columns,
                  loc='center')
plt.axis('off')
plt.tight_layout()
plt.show()
