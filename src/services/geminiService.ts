import {
  GoogleGenerativeAI,
  HarmCategory,
  HarmBlockThreshold,
  GenerateContentRequest,
  GenerationConfig,
} from "@google/generative-ai";
import { AnalysisResult, QuestionResult } from "@/components/StudyAssistant";

// Note: The implementation for extractTextFromPdfPage is assumed to exist in your project.
// import { extractTextFromPdfPage } from "@/utils/pdfReader";

// --- 1. CONFIGURATION ---
// Load the API key from environment variables for security.
const GEMINI_API_KEY = import.meta.env.VITE_GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
  throw new Error("VITE_GEMINI_API_KEY is not set in your environment variables. Please add it to your .env.local file.");
}

// Initialize the Generative AI client and model.
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({
  model: "gemini-1.5-flash-latest",
  // Adjust safety settings if needed for academic content. BLOCK_NONE is permissive.
  safetySettings: [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
  ],
});

// Default generation config to ensure JSON output and consistency.
const defaultGenerationConfig: GenerationConfig = {
    temperature: 0.7,
    maxOutputTokens: 3000,
    responseMimeType: "application/json", // Critical for reliable JSON parsing
};


// --- 2. CORE API HELPER WITH RETRY LOGIC ---
/**
 * A robust wrapper for the Gemini API call that includes retry logic and handles JSON parsing.
 * @param request The content request to send to the model.
 * @param generationConfig Optional override for the generation configuration.
 * @param retries Number of times to retry on failure.
 * @returns The parsed JSON object from the model's response.
 */
const generateContentWithRetry = async (
  request: GenerateContentRequest,
  generationConfig: GenerationConfig = defaultGenerationConfig,
  retries = 3
): Promise<any> => {
  let lastError: Error | undefined;

  for (let i = 0; i < retries; i++) {
    try {
      const result = await model.generateContent({ ...request, generationConfig });
      const responseText = result.response.text();
      return JSON.parse(responseText);
    } catch (error) {
      lastError = error as Error;
      console.error(`Gemini API call attempt ${i + 1} of ${retries} failed:`, error);
      // Wait before retrying (simple backoff)
      if (i < retries - 1) {
        await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
      }
    }
  }
  throw new Error(`Gemini API call failed after ${retries} attempts. Last error: ${lastError?.message}`);
};


// --- 3. EXPORTED SERVICE FUNCTIONS ---

/**
 * Analyzes a single image file for TNPSC exam preparation.
 * @param file The image file to analyze.
 * @param outputLanguage The desired output language ('english' or 'tamil').
 * @returns A promise that resolves to an AnalysisResult object.
 */
export const analyzeImage = async (file: File, outputLanguage: "english" | "tamil" = "english"): Promise<AnalysisResult> => {
  const base64Image = await convertToBase64(file);
  const imagePart = { inlineData: { mimeType: file.type, data: base64Image.split(',')[1] } };

  const languageInstruction = outputLanguage === "tamil" 
    ? "Provide all responses in Tamil language, using Tamil script."
    : "Provide all responses in English language.";

  const prompt = `
Analyze this image for TNPSC exam preparation. ${languageInstruction}
Return a single, valid JSON object based on this exact structure:
{
  "mainTopic": "string",
  "studyPoints": [{"title": "string", "description": "string", "importance": "high|medium|low", "tnpscRelevance": "string", "tnpscPriority": "high|medium|low", "memoryTip": "string"}],
  "keyPoints": ["string"],
  "summary": "string",
  "tnpscRelevance": "string",
  "tnpscCategories": ["string"],
  "difficulty": "easy|medium|hard"
}
Focus on key facts, dates, names, and concepts relevant to TNPSC Group 1, 2, & 4 exams. Make key points and memory tips concise.`;

  const request: GenerateContentRequest = {
      contents: [{ role: "user", parts: [{ text: prompt }, imagePart] }],
  };

  const result = await generateContentWithRetry(request, { ...defaultGenerationConfig, maxOutputTokens: 2048 });
  
  return {
    keyPoints: result.keyPoints || [],
    summary: result.summary || '',
    tnpscRelevance: result.tnpscRelevance || '',
    studyPoints: result.studyPoints || [],
    tnpscCategories: result.tnpscCategories || []
  };
};

/**
 * Generates quiz questions based on one or more analysis results.
 * @param analysisResults An array of analysis results.
 * @param difficulty The desired difficulty of the questions.
 * @param outputLanguage The desired output language.
 * @returns A promise that resolves to a QuestionResult object.
 */
export const generateQuestions = async (
  analysisResults: AnalysisResult[],
  difficulty: string = "medium",
  outputLanguage: "english" | "tamil" = "english"
): Promise<QuestionResult> => {
  const combinedContent = analysisResults.map(result => `Key Points: ${result.keyPoints.join('; ')}\nSummary: ${result.summary}`).join('\n\n');

  const languageInstruction = outputLanguage === "tamil" 
    ? "Generate all questions, options, and explanations in Tamil language."
    : "Generate all questions, options, and explanations in English language.";

  const prompt = `
Based on the following TNPSC study content, generate 15-20 questions.
Difficulty: ${difficulty}.
Language: ${languageInstruction}
Content:\n${combinedContent}

Generate ONLY these types:
- 70% Multiple Choice Questions (MCQ) with 4 options.
- 30% Assertion-Reason questions with 4 standard options.

Return a single, valid JSON array of objects with this exact structure:
[
  {
    "question": "string",
    "options": ["string", "string", "string", "string"],
    "answer": "A|B|C|D", 
    "type": "mcq" | "assertion_reason",
    "difficulty": "${difficulty}",
    "tnpscGroup": "Group 1" | "Group 2" | "Group 4",
    "explanation": "string"
  }
]
CRITICAL: The 'answer' field must be ONLY the capital letter (A, B, C, or D) of the correct option. For Assertion-Reason, use the standard A, B, C, D options.`;

  const request: GenerateContentRequest = {
      contents: [{ role: "user", parts: [{ text: prompt }] }],
  };

  const questions = await generateContentWithRetry(request);

  // Sanitize the output to prevent frontend crashes from unexpected API responses.
  const formattedQuestions = questions.map((q: any) => ({
    ...q,
    type: q.type === "mcq" || q.type === "assertion_reason" ? q.type : "mcq",
    options: Array.isArray(q.options) && q.options.length === 4 ? q.options : ["Option A", "Option B", "Option C", "Option D"],
    answer: ["A", "B", "C", "D"].includes(q.answer) ? q.answer : "A",
    explanation: q.explanation || "No explanation provided."
  }));

  return {
    questions: formattedQuestions,
    summary: analysisResults.map(c => c.summary).join(' '),
    keyPoints: analysisResults.flatMap(r => r.keyPoints),
    difficulty,
    totalQuestions: formattedQuestions.length
  };
};

/**
 * Analyzes the full text content of a PDF, processing pages in parallel batches for performance.
 * @param textContent The full text extracted from a PDF, with pages delimited by markers.
 * @param outputLanguage The desired output language.
 * @returns A promise resolving to a comprehensive analysis object.
 */
export const analyzePdfContentComprehensive = async (
  textContent: string,
  outputLanguage: "english" | "tamil" = "english"
): Promise<{
  pageAnalyses: Array<any>;
  overallSummary: string;
  totalKeyPoints: string[];
  tnpscCategories: string[];
}> => {
  const pageRegex = /==Start of OCR for page (\d+)==([\s\S]*?)==End of OCR for page \1==/g;
  const pageMatches = Array.from(textContent.matchAll(pageRegex));
  console.log(`Found ${pageMatches.length} pages to analyze.`);

  // Process pages concurrently in batches to avoid rate limits and improve speed.
  const BATCH_SIZE = 5; // Number of pages to process in parallel
  const allPageAnalyses: any[] = [];
  
  for (let i = 0; i < pageMatches.length; i += BATCH_SIZE) {
      const batch = pageMatches.slice(i, i + BATCH_SIZE);
      console.log(`Processing batch starting at page ${batch[0]?.[1]}...`);
      
      const promises = batch.map(match => {
          const pageNumber = parseInt(match[1], 10);
          const pageContent = match[2].trim();
          
          if (pageContent.length < 50) return Promise.resolve(null); // Skip empty pages

          // Call the individual page analyzer
          return analyzeIndividualPage(pageContent, pageNumber, outputLanguage)
              .catch(error => {
                  console.error(`Error analyzing page ${pageNumber} in batch:`, error);
                  return null; // Return null on failure to not break Promise.all
              });
      });
      
      const batchResults = await Promise.all(promises);
      allPageAnalyses.push(...batchResults.filter(r => r !== null)); // Add successful results
  }

  const allKeyPoints = allPageAnalyses.flatMap(p => p.keyPoints || []);
  const allCategories = allPageAnalyses.flatMap(p => p.tnpscCategories || []);

  return {
    pageAnalyses: allPageAnalyses,
    overallSummary: `Comprehensive analysis of ${allPageAnalyses.length} pages completed, identifying ${allKeyPoints.length} total key points.`,
    totalKeyPoints: allKeyPoints,
    tnpscCategories: [...new Set(allCategories)]
  };
};

/**
 * Analyzes the text content of a single PDF page. (Internal helper function)
 * @param textContent The text content of the page.
 * @param pageNumber The page number.
 * @param outputLanguage The desired output language.
 * @returns A promise resolving to the page's analysis object.
 */
export const analyzeIndividualPage = async (
  textContent: string,
  pageNumber: number,
  outputLanguage: "english" | "tamil" = "english"
): Promise<any> => {
    const languageInstruction = outputLanguage === "tamil" ? "Provide all responses in Tamil." : "Provide all responses in English.";
    
    const prompt = `
Analyze this content from page ${pageNumber} of a PDF for TNPSC exam preparation. ${languageInstruction}
Content: """${textContent.substring(0, 15000)}"""

Return a single, valid JSON object with this exact structure:
{
  "keyPoints": ["At least 8 short, crisp, memorizable points"],
  "studyPoints": [{"title": "string", "description": "string", "importance": "high|medium|low", "tnpscRelevance": "string"}],
  "summary": "Brief summary of the page content",
  "tnpscRelevance": "How this page's content relates to TNPSC exams",
  "tnpscCategories": ["Category1", "Category2"]
}
Focus on names, dates, places, events, definitions, and key data.`;

    const request: GenerateContentRequest = {
        contents: [{ role: "user", parts: [{ text: prompt }] }],
    };
    
    const analysis = await generateContentWithRetry(request, { ...defaultGenerationConfig, maxOutputTokens: 2500 });

    return {
        pageNumber,
        keyPoints: analysis.keyPoints || [],
        studyPoints: analysis.studyPoints || [],
        summary: analysis.summary || '',
        tnpscRelevance: analysis.tnpscRelevance || '',
        tnpscCategories: analysis.tnpscCategories || []
    };
};

/**
 * Orchestrator function to analyze multiple image files and generate questions.
 * @param files An array of image files.
 * @param difficulty Desired question difficulty.
 * @param outputLanguage Desired output language.
 * @returns A promise resolving to a QuestionResult object.
 */
export const analyzeMultipleImages = async (
  files: File[],
  difficulty: string = "medium",
  outputLanguage: "english" | "tamil" = "english"
): Promise<QuestionResult> => {
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    if (imageFiles.length === 0) {
      throw new Error('No valid images found for analysis');
    }

    // Analyze all images in parallel for maximum performance
    const analysisPromises = imageFiles.map(file => analyzeImage(file, outputLanguage));
    const analysisResults = await Promise.all(analysisPromises);
    
    return generateQuestions(analysisResults, difficulty, outputLanguage);
};


// --- 4. UTILITY FUNCTIONS ---
/**
 * Converts a File object to a base64 encoded string.
 */
const convertToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = error => reject(error);
  });
};