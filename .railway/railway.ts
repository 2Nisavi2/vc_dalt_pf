// Infraestructura como código de Railway (opcional).
// Aplica la configuración del servicio con:  npx railway config plan  ->  npx railway config apply
// El build lo hace el Dockerfile de la raíz; aquí solo se declaran el health check y el nombre del servicio.
// Las credenciales del endpoint NO van en este archivo: se definen como variables del servicio
// (CV_PREDICTION_ENDPOINT y CV_PREDICTION_KEY) en el panel de Railway o con `railway variables --set`.
import { defineRailway, project, service } from "railway/iac";

export default defineRailway(() => {
  const app = service("visionvial", {
    healthcheck: "/_stcore/health",
    healthcheckTimeout: 300,
  });

  return project("visionvial", {
    resources: [app],
  });
});
