# Black-Litterman aplicado a Gestión de Proveedores

Aplicación académica en Python + Streamlit que adapta la lógica del modelo
Black-Litterman a una cartera de proveedores.

## ¿Qué representa cada elemento?

- Activos → proveedores.
- Pesos de mercado → participación actual del gasto/abastecimiento.
- Rendimientos → desempeño mensual del proveedor.
- Views → expectativas del gestor de compras/AP.
- Confianza → nivel de certeza en cada view.
- Optimización → asignación sugerida del abastecimiento.

## Archivos

- `app.py` — aplicación Streamlit.
- `proveedores.csv` — datos sintéticos de ejemplo.
- `requirements.txt` — librerías necesarias.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Después abre la URL local que muestre Streamlit.

## Subir a GitHub

Sube estos tres archivos al mismo nivel:

```text
app.py
proveedores.csv
requirements.txt
```

El `README.md` es opcional pero recomendable.

## Publicar en Streamlit Community Cloud

1. Sube el proyecto a un repositorio de GitHub.
2. En Streamlit Community Cloud selecciona el repositorio.
3. Selecciona `app.py` como archivo principal.
4. Despliega la aplicación.

Los datos son sintéticos y se utilizan únicamente con fines académicos.
