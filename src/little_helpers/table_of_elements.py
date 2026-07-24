# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 09:11:12 2026

@author: southan
"""
import numpy as np
import pandas as pd

# atomic weights from: Prohaska et al., "Standard atomic weights of the
# elements 2021 (IUPAC Technical Report)", Pure and Applied Chemistry 2022,
# 94 (5), 573-600, doi 10.1515/pac-2019-0603.

element_names = {'H': 'hydrogen', 'He': 'helium', 'Li': 'lithium',
                 'Be': 'beryllium', 'B': 'boron', 'C': 'carbon',
                 'N': 'nitrogen', 'O': 'oxygen', 'F': 'fluorine', 'Ne': 'neon',
                 'Na': 'sodium', 'Mg': 'magnesium', 'Al': 'aluminium',
                 'Si': 'silicon', 'P': 'phosphorus', 'S': 'sulfur',
                 'Cl': 'chlorine', 'Ar': 'argon', 'K': 'potassium',
                 'Ca': 'calcium', 'Sc': 'scandium', 'Ti': 'titanium',
                 'V': 'vanadium', 'Cr': 'chromium', 'Mn': 'manganese',
                 'Fe': 'iron', 'Co': 'cobalt', 'Ni': 'nickel', 'Cu': 'copper',
                 'Zn': 'zinc', 'Ga': 'gallium', 'Ge': 'germanium',
                 'As': 'arsenic', 'Se': 'selenium', 'Br': 'bromine',
                 'Kr': 'krypton', 'Rb': 'rubidium', 'Sr': 'strontium',
                 'Y': 'yttrium', 'Zr': 'zirconium', 'Nb': 'niobium',
                 'Mo': 'molybdenum', 'Tc': 'technetium', 'Ru': 'ruthenium',
                 'Rh': 'rhodium', 'Pd': 'palladium', 'Ag': 'silver',
                 'Cd': 'cadmium', 'In': 'indium', 'Sn': 'tin',
                 'Sb': 'antimony', 'Te': 'tellurium', 'I': 'iodine',
                 'Xe': 'xenon', 'Cs': 'caesium', 'Ba': 'barium',
                 'La': 'lanthanum', 'Ce': 'cerium', 'Pr': 'praseodymium',
                 'Nd': 'neodymium', 'Pm': 'promethium', 'Sm': 'samarium',
                 'Eu': 'europium', 'Gd': 'gadolinium', 'Tb': 'terbium',
                 'Dy': 'dysprosium', 'Ho': 'holmium', 'Er': 'erbium',
                 'Tm': 'thulium', 'Yb': 'ytterbium', 'Lu': 'lutetium',
                 'Hf': 'hafnium', 'Ta': 'tantalum', 'W': 'tungsten',
                 'Re': 'rhenium', 'Os': 'osmium', 'Ir': 'iridium',
                 'Pt': 'platinum', 'Au': 'gold', 'Hg': 'mercury',
                 'Tl': 'thallium', 'Pb': 'lead', 'Bi': 'bismuth',
                 'Po': 'polonium', 'At': 'astatine', 'Rn': 'radon',
                 'Fr': 'francium', 'Ra': 'radium', 'Ac': 'actinium',
                 'Th': 'thorium', 'Pa': 'protactinium', 'U': 'uranium',
                 'Np': 'neptunium', 'Pu': 'plutonium', 'Am': 'americium',
                 'Cm': 'curium', 'Bk': 'berkelium', 'Cf': 'californium',
                 'Es': 'einsteinium', 'Fm': 'fermium', 'Md': 'mendelevium',
                 'No': 'nobelium', 'Lr': 'lawrencium', 'Rf': 'rutherfordium',
                 'Db': 'dubniuma', 'Sg': 'seaborgium', 'Bh': 'bohrium',
                 'Hs': 'hassium', 'Mt': 'meitnerium', 'Ds': 'darmstadtium',
                 'Rg': 'roentgenium', 'Cn': 'copernicium', 'Nh': 'nihonium',
                 'Fl': 'flerovium', 'Mc': 'moscovium', 'Lv': 'livermorium',
                 'Ts': 'tennessine', 'Og': 'oganesson'}
element_weights = {'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122,
                   'B': 10.81, 'C': 12.011, 'N': 14.007, 'O': 15.999,
                   'F': 18.998, 'Ne': 20.18, 'Na': 22.99, 'Mg': 24.305,
                   'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06,
                   'Cl': 35.45, 'Ar': 39.95, 'K': 39.098, 'Ca': 40.078,
                   'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996,
                   'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693,
                   'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723, 'Ge': 72.63,
                   'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798,
                   'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224,
                   'Nb': 92.906, 'Mo': 95.95, 'Tc': np.nan, 'Ru': 101.07,
                   'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41,
                   'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6,
                   'I': 126.9, 'Xe': 131.29, 'Cs': 132.91, 'Ba': 137.33,
                   'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24,
                   'Pm': np.nan, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25,
                   'Tb': 158.93, 'Dy': 162.5, 'Ho': 164.93, 'Er': 167.26,
                   'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97, 'Hf': 178.49,
                   'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23,
                   'Ir': 192.22, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59,
                   'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Po': np.nan,
                   'At': np.nan, 'Rn': np.nan, 'Fr': np.nan, 'Ra': np.nan,
                   'Ac': np.nan, 'Th': 232.04, 'Pa': 231.04, 'U': 238.03,
                   'Np': np.nan, 'Pu': np.nan, 'Am': np.nan, 'Cm': np.nan,
                   'Bk': np.nan, 'Cf': np.nan, 'Es': np.nan, 'Fm': np.nan,
                   'Md': np.nan, 'No': np.nan, 'Lr': np.nan, 'Rf': np.nan,
                   'Db': np.nan, 'Sg': np.nan, 'Bh': np.nan, 'Hs': np.nan,
                   'Mt': np.nan, 'Ds': np.nan, 'Rg': np.nan, 'Cn': np.nan,
                   'Nh': np.nan, 'Fl': np.nan, 'Mc': np.nan, 'Lv': np.nan,
                   'Ts': np.nan, 'Og': np.nan}
element_numbers = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7,
                   'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13,
                   'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,
                   'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
                   'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31,
                   'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
                   'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
                   'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49,
                   'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
                   'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61,
                   'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67,
                   'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73,
                   'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79,
                   'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
                   'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
                   'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97,
                   'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102,
                   'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107,
                   'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
                   'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117,
                   'Og': 118}

table_of_elements = pd.DataFrame(
    [], index=pd.Index(element_names.keys(), name='symbol'))
table_of_elements['atomic number'] = pd.Series(element_numbers)
table_of_elements['name'] = pd.Series(element_names)
table_of_elements['atomic weight'] = pd.Series(element_weights)
