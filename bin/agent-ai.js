#!/usr/bin/env node

const { main } = require("../lib/agent-ai-cli");

main(process.argv.slice(2)).catch((error) => {
  const message = error && error.message ? error.message : String(error);
  console.error(`\nAgent AI installer failed: ${message}`);
  process.exitCode = 1;
});
