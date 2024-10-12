import os
from pinecone import Pinecone, ServerlessSpec

# Inicializar Pinecone usando la API Key
pc = Pinecone(
    api_key="587df39e-f0ea-41c7-90b4-3596c9d1a3ca"
)

# Verificar la lista de índices para confirmar que la conexión es exitosa
indexes = pc.list_indexes()

if indexes:
    print("Conexión exitosa. Índices disponibles:", indexes)
else:
    print("No se encontraron índices. Conexión a Pinecone es exitosa, pero no hay índices disponibles.")
