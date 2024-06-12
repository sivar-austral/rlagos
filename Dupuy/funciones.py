import matplotlib.patheffects as path_effects
from collections import Counter 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np
import os 
import re

def extraccion_rutas(variables, casos, carpeta):

    stack= {key: '' for key in variables} #dic. con toda la información
    for var in variables: 
        stack_casos= {key: '' for key in casos}
        
        for caso in casos: 
            ruta_caso    = carpeta + var + "/" + caso 
            listado_files= sorted( os.listdir(ruta_caso) )
            numpy_files  = [archivo for archivo in listado_files if archivo.endswith('.npy')]

            if len(numpy_files)== 242: 
                numpy_files.pop(-1)
                numpy_files.pop(120)

            numpy_files.pop(120)
            numpy_files.pop(0)

        #Check 
            patron     = re.compile(r'(\d{2}):')

            primer_d03 = numpy_files[0]
            ultimo_d03 = numpy_files[118]
            primer_d05 = numpy_files[119]
            ultimo_d05 = numpy_files[237]

            patron_d03_01 = patron.search(primer_d03).group(1)
            patron_d05_01 = patron.search(primer_d05).group(1)
            patron_d03_119 = patron.search(ultimo_d03).group(1)
            patron_d05_119 = patron.search(ultimo_d05).group(1)

            if not patron_d03_01 == patron_d05_01 and patron_d03_119 == patron_d05_119:
                print("Función interrumpida")
                break 

            parches_npy=[] 

            for archivo in numpy_files: 
                ruta  =os.path.join(ruta_caso, archivo)
                parche= np.load(ruta)
                parches_npy.append(parche)

            if var != 'times': #la variable times requiere un tratamiento especial al ser una ctte.
                stack_casos[caso] = tf.squeeze(tf.convert_to_tensor(parches_npy))
            else:
                stack_casos[caso] = parches_npy

        print(f"variable {var} lista")

        stack[var] = stack_casos #guardamos el dic con los casos en su respectiva key   

    return stack     


def calcular_frecuencias(etiquetas_cuad_testeo, indices, idx_cuadrantes):

    cuadrantes= [ int(etiquetas_cuad_testeo[idx][1][0]) for idx in indices]
    counter_cuad= Counter(cuadrantes)
    freq_cuadrantes= list(counter_cuad.items())

    
    for cuad in idx_cuadrantes:
        freq_cuadrantes.append( [cuad,0])

    freq_cuadrantes=  sorted(freq_cuadrantes, key=lambda x: x[0])
    return freq_cuadrantes, counter_cuad


def grafico_espacial(freq_cuad, counter_cuad, str_conjunto='', cmap='viridis'): 


    data = np.random.random((12, 13))
    for idx in freq_cuad:
        fila = idx[0] // data.shape[1]
        columna = idx[0] % data.shape[1]
        data[fila,columna] = idx[1]


    #Etiquetas
    xlabs=[]; ylabs=[]
    for i in range(13):
        xlabs.append(str(i))

    for j in range(12):
        ylabs.append(str(j))

         
    #Mapa de calor
    fig, ax = plt.subplots(figsize=(18, 8))
    im = ax.imshow(data, cmap=cmap)

    
    ax.set_xticks(np.arange(len(xlabs)), labels = xlabs)
    ax.set_yticks(np.arange(len(ylabs)), labels = ylabs)
    ax.set_title(f"Parches por cuadrante del conjunto de {str_conjunto}", fontsize=10)


    for i in range(len(ylabs)  ):
        for j in range(len(xlabs)  ):
                text = ax.text(j, i, int(data[i, j]),
                          ha = "center", va = "center", color = "w", fontsize=10, 
                          path_effects=[path_effects.withStroke(linewidth=1, foreground='black')])
        
    #Agregar la leyenda
    cbar = ax.figure.colorbar(im, ax = ax)
    cbar.ax.set_ylabel("Frecuencias", rotation = -90, va = "top")
    vmax=counter_cuad.most_common()[1][1]
    vmin=counter_cuad.most_common()[-1][1]
    im.set_clim(vmin,vmax)

    plt.show() 