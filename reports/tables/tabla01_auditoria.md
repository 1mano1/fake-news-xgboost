|   Paso | Regla aplicada                                         |   Filas eliminadas |   Filas resultantes |   Real |   Fake |
|-------:|:-------------------------------------------------------|-------------------:|--------------------:|-------:|-------:|
|      0 | Conjunto original                                      |                  0 |               72134 |  35028 |  37106 |
|      1 | Eliminar filas con title y text nulos/vacíos           |                  0 |               72134 |  35028 |  37106 |
|      2 | Eliminar text con < 20 caracteres                      |                881 |               71253 |  35025 |  36228 |
|      3 | Eliminar duplicados exactos (title + text)             |               8225 |               63028 |  34787 |  28241 |
|      4 | Eliminar cuerpos idénticos con etiqueta contradictoria |                  0 |               63028 |  34787 |  28241 |
|      5 | Eliminar duplicados por text idéntico                  |                395 |               62633 |  34617 |  28016 |