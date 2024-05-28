import matplotlib.pyplot as plt
import numpy as np 
import random 

#parches=np.zeros((93600, 32, 32))

class Data_split: 

    def __init__(self, parches, indices=None, seed=None):

        self.parches=parches      #[tensor] Recibe el tensor con los parches
        self.seed=seed            #[int]    Semilla para la extracción aleatoria de muestras
        self.indices=indices      #[List]   Índices de parches a conservar 
        self.etiquetas_cuadrantes= np.zeros((self.parches.shape[0], 1)) #Array que contendrá el id del cuadrante al que pertenece cada parche
        self.etiquetas_tiempos   = np.zeros((self.parches.shape[0], 1)) #Array que contendrá el id del tiempo al que pertenece cada parche 
        self.cant_cuadrantes=156  #Cantidad de cuadrantes.
        self.cant_tiempos= int(parches.shape[0] / 156)  #Cantidad de registros temporales distintos
             
        if self.seed is not None:
            random.seed(self.seed) 

    def init_etiquetas(self):

        ''' Definición de las variables de clase que contendrán las 
        etiquetas del cuadrante y tiempo al que pertenece cada parche i.
        En caso de que la variable self.indice sea None, se trabajará con todos los parches. 
        
        Nota: self.etiquetas_cuadrantes: Cuadrante {0,1,...,155} al que pertenece el parche i
              self.etiquetas_tiempos: Tiempo {0,1,2,...,599} al que pertenece el parche i
        '''  
        #cantidad de cuadrantes dado los parches de tamaño 32x32
        for cuad in range(self.cant_cuadrantes):
            idx= list( range(cuad,self.parches.shape[0],156) )
            self.etiquetas_cuadrantes[idx]= cuad

        for t in range(self.cant_tiempos):
            self.etiquetas_tiempos[156*t:156*(t+1)]=t

        if self.indices is None: 

            self.etiquetas_cuadrantes= list ( enumerate ( self.etiquetas_cuadrantes))
            self.etiquetas_tiempos   = list ( enumerate ( self.etiquetas_tiempos   ))
            self.etiquetas = [idx for idx, _ in self.etiquetas_cuadrantes]

        else:
            self.etiquetas_cuadrantes=self.etiquetas_cuadrantes[self.indices]
            self.etiquetas_tiempos=self.etiquetas_tiempos[self.indices]
            self.etiquetas_cuadrantes= list(zip(self.indices , self.etiquetas_cuadrantes))
            self.etiquetas_tiempos   = list(zip(self.indices, self.etiquetas_tiempos))
            self.etiquetas= self.indices 
            
        print(f" {len(self.etiquetas_cuadrantes)} etiquetas espaciales y temporales creadas correctamente")


        
    def intervalo_temporal(self, porc_intervalo=0.05,indices=None, verbose=False):

        ''' Método encargado de extraer un intervalo continuo de valores de tiempo (self.tiempo_intervalo)
            de la variable idx_tiempos. La variable idx_tiempos contiene inicialmente todos los valores posibles 
            de tiempo. Si índice es None, se trabajarán con todos los registros temporales. 
            
            Nota: invariante a la cantidad de parches.

        input: 
            porc_intervalo: [float]  porcentaje ]0,1[ de tiempos a extraer 
            verbose: [Booleano] si True, imprime información relacionada al intervalo extraído 
            
        return:
            self.tiempo_intervalo: [Lista] Valores del intervalo de tiempos extraído. 
        '''

        self.idx_tiempos= list(range(self.cant_tiempos)) 
        if self.indices is not None:
            self.idx_tiempos=[idx for idx in self.idx_tiempos if idx not in indices]
        len_intervalo= int(self.cant_tiempos * porc_intervalo) #invariable
        idx_inicio = random.randint(0, len(self.idx_tiempos) - len_intervalo )
        self.tiempo_intervalo = self.idx_tiempos[idx_inicio: idx_inicio + len_intervalo] 
        self.idx_tiempos = [ valor for  valor in self.idx_tiempos if valor not in self.tiempo_intervalo]

        if verbose:
            print("Proceso 1: Extracción intervalo temporal")
            print("Cantidad de tiempos extraidos: ", len(self.tiempo_intervalo))
            print("Cantidad de tiempos restantes: ", len(self.idx_tiempos))
            print("-"*60)

        return self.tiempo_intervalo
    
    def muestra_temporal(self, porc_muestra=0.1, verbose=False): 

        ''' Método encargado de extraer una muestra aleatoria: tiempo_muestra, de tiempos de 
        la variable self.idx_tiempos. 
        
        Nota: Invariante respecto a la cantidad de parches.

        input: 
        porc_muestra: [float] porcentaje ]0.1[ que define el tamaño de la muestra a extraer. 
        verbose: [Booloeano] si True, imprime información relacionada a la muestra extraída. 

        return:
            self.tiempo_muestra: [lista] Valores de tiempo de la muestra extraída. 
        '''

        len_muestra= int( self.cant_tiempos * porc_muestra)
        self.tiempo_muestra = random.sample(self.idx_tiempos, len_muestra )
        self.idx_tiempos = [ idx for idx in self.idx_tiempos if idx not in self.tiempo_muestra]

        if verbose:
            print("Proceso 2: Extracción muestra temporal")
            print("Cantidad de tiempos extraidos: ", len(self.tiempo_muestra))
            print("Cantidad de tiempos restantes: ", len(self.idx_tiempos))
            print("-"*60)
            
        return self.tiempo_muestra 
    
    def extraccion_temporal(self, verbose=True):

        ''' Método encargado de extraer los parches cuya etiqueta coincide con los tiempos extraídos:
        tiempo_extraer, en los métodos anteriores. 

        input: 
            verbose: [Booleano] si True, imprime información de la cantidad de parches extraídos.

        return:
            idx_extraer= valores de los tiempos a extraer.
        '''

        self.tiempo_extraer= self.tiempo_intervalo + self.tiempo_muestra 
        self.idx_parches_tiempo= [ indice for indice, valor in self.etiquetas_tiempos if valor not in self.tiempo_extraer]
        
        if verbose: 

            print("Proceso 3: Exclusión de parches dado etiquetado temporal")
            print("Cantidad de tiempos extraídos totales: ",len(self.tiempo_extraer))
            print("Cantidad de parches original: ", len(self.etiquetas_tiempos))
            print("Cantidad de parches extraídos: ", len(self.etiquetas_tiempos) - len(self.idx_parches_tiempo)  )
            print("Cantidad de parches restantes: ", len(self.idx_parches_tiempo))
            print("-"*60)

        return self.tiempo_extraer

    def muestra_espacial(self, porc_cuadrantes=0.05, indices= None, verbose=False):

        ''' Método encargado de extraer una muestra aleatoria de cuadrantes de la variable: self.idx_cuadrantes
        que contiene todos los valores de los cuadrantes. Si indices is None, se trabaja con todos los cuadrantes.

        input:
            poc_cuadrantes: [float] Porcentaje ]0,1[ que define el tamaño de la muestra a extraer.
            verbose: [Booleano] si True, imprime información relacionada a los cuadrantes extraídos

        return:
            self.cuad_muestra
        '''

        self.idx_cuadrantes=list(range(self.cant_cuadrantes))
        if indices is not None:
            self.idx_cuadrantes= [idx for idx in self.idx_cuadrantes if idx not in indices]
        self.cuad_muestra= random.sample(self.idx_cuadrantes, int(self.cant_cuadrantes * porc_cuadrantes))  
        self.idx_cuadrantes= [cuad for cuad in self.idx_cuadrantes if cuad not in self.cuad_muestra]

        if verbose:
            print("Proceso 4: Extracción de cuadrantes")
            print("Cantidad de cuadrantes extraidos: ", len(self.cuad_muestra))
            print("Cantidad de cuadrantes restantes: ", len(self.idx_cuadrantes))
            print("-"*60)

        return self.cuad_muestra
    
    def extraccion_espacial(self, verbose=False):

        ''' Método encargado de extraer todos los parches cuya etiqueta coincide con los cuadrantes
        extraidos en el método anterior.

        input:
            verbose: [Booleano] si True, imprime información relacionada a los cuadrantes extraídos
        
        return:
            self.idx_parches_mantener
        '''

        self.idx_parches_cuad_extraer   = [ indice for indice, valor in self.etiquetas_cuadrantes if valor in self.cuad_muestra] 
        self.idx_parches_mantener= list( set(self.idx_parches_tiempo) -  set(self.idx_parches_cuad_extraer) ) 

        if verbose:
            print("Proceso 5: Exclusión de parches dado etiquetado cuadrante.")
            print("Cantidad de parches extraídos: ", len(self.idx_parches_cuad_extraer))
            print("Cantidad de parches posterior extracción espacial: ", len(self.idx_parches_mantener))
            print("-"*60)
        
        return self.idx_parches_mantener 
    
    def extraccion_final(self, porc_muestra=0.1, verbose=False):

        ''' Método encargado de extraer una muestra aleatoria de parches.

        input: 
            poc_cuadrantes: [float] Porcentaje ]0,1[ que define el tamaño de la muestra a extraer.
            verbose: [Booleano] si True, imprime información relacionada a los cuadrantes extraídos

        return:
            self.idx_parches_mantener 
            self.idx_parches_extraer
        '''

        self.len_parches_mantener=len(self.idx_parches_mantener)
        self.idx_parches_mantener= random.sample(self.idx_parches_mantener, int(  len(self.idx_parches_mantener) * ( 1 - porc_muestra) ) )
        self.idx_parches_extraer= list (set(self.etiquetas) - set(self.idx_parches_mantener))
        
        if verbose:
            print("Proceso 6: Extracción de muestra aleatoria.")
            print("Cantidad de parches extraídos: ", self.len_parches_mantener - len(self.idx_parches_mantener))
            print("cantidad de parches restantes: ", len(self.idx_parches_mantener))
            print("-"*60)
            print("Resumen: ")
            print("Cantidad de parches restantes final: ", len(self.idx_parches_mantener) )
            print("Cantidad de parches extraídos final: ", len(self.idx_parches_extraer) )

        return self.idx_parches_mantener, self.idx_parches_extraer




def grafico_temporal(indices_extraer):


    cant_tiempos=600
    tiempos=np.zeros(cant_tiempos)
    tiempos[indices_extraer] = 1
    indices=list(range(cant_tiempos))
    x= np.array(indices)[indices_extraer]
    y= tiempos[indices_extraer]

    plt.figure(figsize=(20, 2))
    plt.stem(
        x, y, linefmt="#193737", markerfmt=' ', basefmt=' ')  
    plt.gca().set_yticks([])


    etiquetas = []
    for dia in range(1, 6):  # Días del 1 al 5
        for subdia in range(0, 5):  
            etiqueta = f"Día: {dia}.{subdia}"
            etiquetas.append(etiqueta)

    etiquetas.append("Fin")

    intervalo = 24
    marcas_x = range(0, len(tiempos) + intervalo, intervalo)
    plt.xticks(marcas_x, etiquetas, rotation='vertical') 


    plt.title(f"Cantidad de tiempos extraídos {len(indices_extraer)} de {cant_tiempos} ")