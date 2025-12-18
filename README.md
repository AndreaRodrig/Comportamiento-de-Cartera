# Modelo de Cobranza para Retail – Predicción de Clientes Buenos y Malos

Este proyecto implementa un **modelo predictivo de cobranza** para clientes de una empresa retail, con el objetivo de identificar **clientes que probablemente se vuelvan “malos” en el próximo mes**. La clasificación “Bueno/Malo” se basa en la metodología LiSim:

- **Buenos**: Clientes cuya obligación **mantiene o mejora su tramo de mora**, y que **no son reestructurados, castigados o adjudicados** en el mes siguiente.  
- **Malos**: Clientes cuya obligación **incrementa su tramo de mora**, se mantiene en mora ≥ 120 días, o **es reestructurada, castigada o adjudicada** en el mes siguiente.

El proyecto se centra en la predicción de **t → t+1**, es decir, el target se calcula para el próximo mes, asegurando **validez temporal** y evitando fugas de información futura.

---

## Objetivos del modelo

1. **Predecir clientes con alto riesgo de mora o deterioro** para priorizar las gestiones de cobranza.  
2. **Optimizar la asignación de recursos** de recuperación, enfocando esfuerzos en los clientes más críticos.  
3. **Proporcionar métricas interpretables para negocio**, como AUC, lift por decil y distribución de malos/buenos.

---

## Fuentes de datos

El modelo integra múltiples fuentes:

- **Información sociodemográfica**: edad, sexo, profesión, tipo de vivienda.  
- **Información económica**: salario, actividad económica, antigüedad laboral.  
- **Información de producto y pagos**: tipo de producto, saldo, cuotas, historial de pagos.  
- **Comportamiento de cartera**: altura de mora, estado de la cartera, reestructuras y castigos.  
- **Gestiones de cobranza**: tipo y resultado de las gestiones realizadas.  
- **Buró de crédito**: información externa de historial crediticio.  

