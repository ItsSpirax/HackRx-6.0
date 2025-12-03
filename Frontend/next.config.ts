import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  env: {
    NEXT_PUBLIC_API_URL: "https://adithvm.centralindia.cloudapp.azure.com",
  },
};

export default nextConfig;
