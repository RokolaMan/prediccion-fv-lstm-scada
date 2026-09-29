# Datos

Los datos del prototipo no se incluyen en el repositorio; están disponibles a solicitud de los autores.

Coloca aquí el archivo `solar_real.csv`. Debe tener, como mínimo, estas columnas:

| Columna       | Descripción                          | Unidad |
|---------------|--------------------------------------|--------|
| `timestamp`   | Fecha y hora de la medición          | —      |
| `voltaje_v`   | Voltaje del panel                    | V      |
| `corriente_a` | Corriente del panel                  | A      |
| `temp_c`      | Temperatura ambiente                 | °C     |
| `hum_pct`     | Humedad relativa                     | %      |
| `lux`         | Iluminancia                          | lx     |
| `potencia_w`  | Potencia generada (variable objetivo)| W      |

Columnas adicionales (`device_id`, `status`, `energia_kwh`, etc.) se ignoran.
