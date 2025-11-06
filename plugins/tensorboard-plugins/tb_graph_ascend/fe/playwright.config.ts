import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 60 * 1000,
  expect: {
    timeout: 5000
  },
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  /* Opt out of parallel tests on CI. */
  workers: 1,
  reporter: 'html',
  use: {
    /* Base URL to use in actions like `await page.goto('')`. */
    baseURL: 'http://localhost:8080',
    trace: 'on-first-retry',
    headless: true
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1920, height: 1080 }
      }
    }
  ],

  /* Run your local dev server before starting the tests */
  // 通过tensorboard指令拉起后端，需确保运行的环境能够执行该指令
  webServer: [
    { 
      command: 'tensorboard --logdir ~/GUI_DATA',
      reuseExistingServer: !process.env.CI
    },
    {
      command: 'npm run dev',
      url: 'http://127.0.0.1:8080',
      reuseExistingServer: !process.env.CI
    }
  ]
});
